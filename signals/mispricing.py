"""
Mispricing signal: model IV vs market IV gap detection.

Buffett margin of safety gate: only flag a mispricing when the vol gap exceeds
config.MIN_VOL_GAP (1.5 vol points) after accounting for calibration uncertainty
and estimated transaction cost. Named in Buffett's honour — we only trade when
the market is clearly wrong by enough to survive our own errors.

Signal anatomy per option:
  gap = model_iv - market_iv
  Positive gap: model says the option is CHEAP (market under-pricing vol)
    → buy the option, delta-hedge long gamma
  Negative gap: model says the option is EXPENSIVE (market over-pricing vol)
    → sell the option, delta-hedge short gamma

Only gaps with |gap| ≥ MIN_VOL_GAP AND Feller-valid parameters get through.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from calibration.heston import HestonParams, heston_implied_vols, feller_satisfied
from data.surface import VolSurface
import config


@dataclass
class MispricingSignal:
    """A single actionable mispricing at one (strike, maturity) point."""
    ticker: str
    snapshot_time: str

    strike: float
    maturity: float               # years
    log_moneyness: float          # log(K/F)

    market_iv: float              # from VolSurface interpolation
    model_iv: float               # from Heston pricer with calibrated params
    vol_gap: float                # model_iv - market_iv

    direction: str                # "buy" (gap>0, option cheap) or "sell" (gap<0, expensive)
    params: HestonParams          # calibrated params used to compute model_iv


@dataclass
class SurfaceMispricing:
    """All mispricings detected on one options surface snapshot."""
    ticker: str
    snapshot_time: str
    params: HestonParams

    signals: list[MispricingSignal]

    # Calibration quality metadata
    calibration_rmse: float
    feller_ok: bool

    @property
    def n_buy(self) -> int:
        return sum(1 for s in self.signals if s.direction == "buy")

    @property
    def n_sell(self) -> int:
        return sum(1 for s in self.signals if s.direction == "sell")

    @property
    def max_abs_gap(self) -> float:
        if not self.signals:
            return 0.0
        return max(abs(s.vol_gap) for s in self.signals)

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "snapshot_time": self.snapshot_time,
            "n_signals": len(self.signals),
            "n_buy": self.n_buy,
            "n_sell": self.n_sell,
            "max_abs_gap": self.max_abs_gap,
            "calibration_rmse": self.calibration_rmse,
            "feller_ok": self.feller_ok,
        }


def compute_model_ivs(
    surface: VolSurface,
    params: HestonParams,
) -> np.ndarray:
    """Compute Heston model IVs at every point on the market surface grid.

    Returns array of model IVs, aligned with surface.market_ivs.
    NaN entries: IV solver failed for that (K, T) point.
    """
    import jax.numpy as jnp

    chain = surface.chain
    S = chain.spot
    r = chain.r
    q = chain.q

    F = S * np.exp((r - q) * surface.maturities)
    strikes_arr = F * np.exp(surface.log_moneyness)

    model_ivs_jax = heston_implied_vols(
        S,
        jnp.array(strikes_arr),
        jnp.array(surface.maturities),
        r,
        q,
        params,
    )
    return np.array(model_ivs_jax)


def detect_mispricings(
    surface: VolSurface,
    params: HestonParams,
    calibration_rmse: float,
    min_vol_gap: float = config.MIN_VOL_GAP,
    cal_t_min: Optional[float] = None,
    cal_t_max: Optional[float] = None,
) -> SurfaceMispricing:
    """Compare Heston model IVs against market IVs. Flag gaps above the Buffett gate.

    Args:
        surface: Market VolSurface from cleaner output.
        params: Calibrated Heston parameters.
        calibration_rmse: Calibration fit quality (in log-IV space). Used to warn
            if calibration is poor (signals unreliable when RMSE is high).
        min_vol_gap: Minimum |model_iv - market_iv| to flag as actionable.
            Default: config.MIN_VOL_GAP (1.5 vol points). The Buffett gate.

    Returns:
        SurfaceMispricing with all actionable signals for this snapshot.
    """
    chain = surface.chain
    feller_ok = feller_satisfied(params)

    # If Feller is violated, calibration is unreliable — return empty signal set
    if not feller_ok:
        return SurfaceMispricing(
            ticker=chain.ticker,
            snapshot_time=chain.snapshot_time,
            params=params,
            signals=[],
            calibration_rmse=calibration_rmse,
            feller_ok=False,
        )

    # Fit-quality gate: a poor fit (RMSE above the Popper threshold) means model IVs are
    # unreliable everywhere, so every gap is a calibration artifact, not a mispricing.
    # Suppress ALL signals — this is what stops a failed fit (e.g. RMSE 0.19) from
    # flooding thousands of garbage signals.
    if calibration_rmse > config.SIGNAL_MAX_RMSE:
        return SurfaceMispricing(
            ticker=chain.ticker,
            snapshot_time=chain.snapshot_time,
            params=params,
            signals=[],
            calibration_rmse=calibration_rmse,
            feller_ok=feller_ok,
        )

    model_ivs = compute_model_ivs(surface, params)
    market_ivs = surface.market_ivs

    F = chain.spot * np.exp((chain.r - chain.q) * surface.maturities)
    strikes_arr = F * np.exp(surface.log_moneyness)

    signals = []
    for i in range(len(market_ivs)):
        market_iv = float(market_ivs[i])
        model_iv = float(model_ivs[i]) if not np.isnan(model_ivs[i]) else np.nan

        if np.isnan(market_iv) or np.isnan(model_iv):
            continue

        # Only signal within the calibrated maturity range (avoid extrapolated tenors).
        if cal_t_min is not None:
            T_i = float(surface.maturities[i])
            if T_i < cal_t_min - 1e-9 or T_i > cal_t_max + 1e-9:
                continue

        # Skip deep ITM/OTM strikes — IV is numerically unstable and vega ~ 0 there,
        # so any "gap" is meaningless and untradeable.
        if abs(float(surface.log_moneyness[i])) > config.MAX_SIGNAL_LOG_MONEYNESS:
            continue

        gap = model_iv - market_iv
        # Below MIN_VOL_GAP: no edge. Above MAX_VOL_GAP: implausible -> IV artifact.
        if abs(gap) < min_vol_gap or abs(gap) > config.MAX_VOL_GAP:
            continue

        signals.append(MispricingSignal(
            ticker=chain.ticker,
            snapshot_time=chain.snapshot_time,
            strike=float(strikes_arr[i]),
            maturity=float(surface.maturities[i]),
            log_moneyness=float(surface.log_moneyness[i]),
            market_iv=market_iv,
            model_iv=model_iv,
            vol_gap=gap,
            direction="buy" if gap > 0 else "sell",
            params=params,
        ))

    # Sort by absolute gap descending — largest mispricings first
    signals.sort(key=lambda s: abs(s.vol_gap), reverse=True)

    return SurfaceMispricing(
        ticker=chain.ticker,
        snapshot_time=chain.snapshot_time,
        params=params,
        signals=signals,
        calibration_rmse=calibration_rmse,
        feller_ok=feller_ok,
    )


def signal_report(sm: SurfaceMispricing) -> str:
    """Human-readable report of detected mispricings."""
    lines = [
        f"{'='*60}",
        f"Mispricing Report: {sm.ticker} at {sm.snapshot_time}",
        f"  Calibration RMSE: {sm.calibration_rmse:.4f}  "
        f"Feller: {'OK' if sm.feller_ok else 'VIOLATED'}",
        f"  Signals: {len(sm.signals)} total ({sm.n_buy} buy, {sm.n_sell} sell)",
        f"  Max |gap|: {sm.max_abs_gap:.4f} ({sm.max_abs_gap*100:.1f} vol pts)",
    ]

    if not sm.signals:
        lines.append("  [no actionable mispricings above Buffett gate]")
        return "\n".join(lines)

    if sm.calibration_rmse > config.CALIBRATION_FAIL_RMSE:
        lines.append(
            f"  [!] RMSE {sm.calibration_rmse:.4f} exceeds Popper threshold "
            f"{config.CALIBRATION_FAIL_RMSE}. Signals unreliable."
        )

    lines.append(f"  {'K':>8} {'T':>6} {'k':>7} {'mkt_iv':>8} {'mdl_iv':>8} {'gap':>8} {'dir':>5}")
    lines.append(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    for s in sm.signals:
        lines.append(
            f"  {s.strike:8.2f} {s.maturity:6.3f} {s.log_moneyness:7.4f} "
            f"{s.market_iv:8.4f} {s.model_iv:8.4f} {s.vol_gap:+8.4f} {s.direction:>5}"
        )

    return "\n".join(lines)

# Reliable 5-minute scheduling (external trigger → workflow_dispatch)

GitHub Actions `schedule:` is best-effort — runs are delayed and dropped under load,
so it cannot guarantee a 5-minute cadence. We trigger the loop from a **reliable
external scheduler** that calls the GitHub API's `workflow_dispatch` endpoint every
5 minutes during market hours. The in-repo `schedule:` cron is kept as a free backup;
the workflow's `concurrency` group serialises runs so a double-fire can never run two
ticks at once.

The loop's own market-hours guard (authoritative via the Alpaca clock) makes any fire
outside regular trading hours a cheap no-op, so the scheduler can fire a wide UTC window
that covers both DST regimes and let the loop decide whether to actually trade.

---

## 1. Create a GitHub token (fine-grained PAT)

GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate new token.
- **Resource owner:** AliAlpOezer
- **Repository access:** Only select repositories → `heston-arb`
- **Permissions:** Repository → **Actions: Read and write** (everything else: No access)
- **Expiration:** as long as you're comfortable (e.g. 90 days; rotate on expiry)

Copy the token (`github_pat_…`). Treat it as a secret — it can trigger workflows.

---

## 2. The request to schedule

```
POST https://api.github.com/repos/AliAlpOezer/heston-arb/actions/workflows/302012980/dispatches
Headers:
  Authorization: Bearer github_pat_XXXXXXXX
  Accept: application/vnd.github+json
  X-GitHub-Api-Version: 2022-11-28
Body (raw JSON):
  {"ref":"master"}
```

(`302012980` is the live-tick workflow id; `.github/workflows/tick.yml` also works in
the URL in place of the id.)

A successful dispatch returns **HTTP 204 No Content** (empty body).

### Test it from your machine first
Type this in the Claude Code prompt (the `!` runs it in-session) or any shell:
```
curl -i -X POST \
  -H "Authorization: Bearer github_pat_XXXXXXXX" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/AliAlpOezer/heston-arb/actions/workflows/302012980/dispatches \
  -d '{"ref":"master"}'
```
Expect `HTTP/2 204`. Then check the Actions tab for a new `workflow_dispatch` run.

---

## 3. Configure cron-job.org (free, reliable)

Create a job at https://cron-job.org → "Create cronjob":
- **URL:** `https://api.github.com/repos/AliAlpOezer/heston-arb/actions/workflows/302012980/dispatches`
- **Request method:** POST
- **Request body:** `{"ref":"master"}`
- **Headers** (Advanced → Headers):
  - `Authorization: Bearer github_pat_XXXXXXXX`
  - `Accept: application/vnd.github+json`
  - `X-GitHub-Api-Version: 2022-11-28`
  - `Content-Type: application/json`
- **Schedule** (custom):
  - Minutes: `*/5`
  - Hours: `13,14,15,16,17,18,19,20,21`  (UTC — covers summer RTH 13:30–20:00 and winter 14:30–21:00)
  - Days of month: every
  - Months: every
  - Days of week: `Mon,Tue,Wed,Thu,Fri`
  - **Timezone: UTC** (important — set the job's timezone to UTC, not local)
- **Notifications:** enable "notify on failure" so you hear if dispatch starts 4xx-ing
  (e.g. token expired).

Save. From the next 5-minute boundary it will POST a dispatch; the loop no-ops until the
market actually opens, then starts ticking — i.e. the first real tick lands at the open.

---

## 4. Alternative: GCP Cloud Scheduler

```
gcloud scheduler jobs create http heston-tick \
  --schedule="*/5 13-21 * * 1-5" \
  --time-zone="UTC" \
  --uri="https://api.github.com/repos/AliAlpOezer/heston-arb/actions/workflows/302012980/dispatches" \
  --http-method=POST \
  --headers="Accept=application/vnd.github+json,X-GitHub-Api-Version=2022-11-28,Authorization=Bearer github_pat_XXXXXXXX" \
  --message-body='{"ref":"master"}'
```

---

## Notes & caveats

- **Runtime vs cadence:** each GHA run reinstalls JAX/Diffrax (~3–4 min). With a 5-min
  cadence the `concurrency` group queues an overlapping fire rather than running two at
  once, so at worst a tick is skipped, never doubled. If consistent sub-5-min ticks
  matter, move to a persistent `run_live` process (deps installed once, ticks in seconds).
- **Token expiry** is the most likely future failure. cron-job.org failure alerts + the
  workflow's own `if: failure()` Telegram alert will surface it.
- **Keep or drop the in-repo `schedule:`** in `tick.yml`: keeping it is a free backup and
  cannot cause a double tick (concurrency). No change needed either way.

# Phase 3 and Phase 4 Plan

Current state: Phase 1 is complete, Phase 2 is scaffolded and ready for an RTX
5090 run, and the repository now has a mock GPU path so the full automation can
be rehearsed before renting hardware.

## Phase 3 -- Result Automation

Goal: turn a queued PR's GPU result into a deterministic public verdict.

### Inputs

- `dashboard/data.json`: live oldest-first queue written by `eval.pr_bot`.
- `gpu-results/pr-<number>-<sha>.json`: wrapped result artifact written by
  `eval.gpu_batch`.
- `eval/ledger.jsonl`: append-only verified result ledger.

### Real RTX 5090 Flow

1. `PR bot queue` runs all day and labels clean PRs `status:queued-gpu`.
2. Maintainer starts `GPU batch evaluation` manually on a self-hosted runner
   labeled `self-hosted`, `gpu`, and `rtx-5090`.
3. `eval.gpu_batch --run --clean` evaluates queued PRs sequentially.
4. Each result is written to `gpu-results/`.
5. `eval.result_bot --write` parses the result, decides `eval:*`, removes
   `status:queued-gpu`, posts a score comment, and appends `eval/ledger.jsonl`.
6. `dashboard/results.json` is rebuilt from the ledger.
7. The workflow commits ledger/result-dashboard changes back to `main`.

### Mock Flow

Use `Mock GPU batch evaluation` before renting hardware:

1. Open a PR that passes non-GPU gates.
2. Wait for the PR bot to add `status:queued-gpu`.
3. Run the mock workflow with `write_results=false` first.
4. Inspect the artifact and logs.
5. Run again with `write_results=true` to apply labels/comments and commit mock
   ledger/dashboard data.

The mock result uses the same JSON wrapper as the real RTX 5090 result. It does
not prove GPU runtime or memory behavior; it proves queue consumption, verdict
labeling, comments, ledger idempotency, and dashboard updates.

### Merge and Close Policy

- The bot may apply labels, comments, ledger rows, and dashboard data.
- The bot may close `eval:REJECT` only when `--close-rejected` is explicitly set.
- The bot should not auto-merge accepted PRs until the system has survived real
  submissions and maintainer review.

### Phase 3 Exit Criteria

- Mock workflow can process a queued PR end-to-end.
- Real RTX 5090 workflow can process one queued PR and produce a valid result.
- Re-running the same result is idempotent: no duplicate ledger rows or comments.
- Final labels match `eval.label.label()` for the measured score.

## Phase 4 -- Public Dashboard

Goal: make the project state understandable from GitHub Pages without reading
workflow logs.

### Data Feeds

- `dashboard/data.json`: live queue and open PR states.
- `dashboard/results.json`: verified result projection from `eval/ledger.jsonl`.

### Next.js App

The dashboard is implemented under `dashboard/` using Next.js static export:

- queue count and oldest-first PR order
- open PR tracking state
- verified result table
- admitted/rejected counts
- frontier score by track
- RTX 5090 evaluation pin

### Publishing

`Dashboard Pages` builds the Next.js static export on pushes touching
`dashboard/**` and deploys `dashboard/out` through GitHub Pages.

### Phase 4 Exit Criteria

- GitHub Pages shows queue and verified results after bot/result updates.
- Dashboard rebuilds without needing GPU hardware.
- Miner-facing docs explain how to run local tests and what labels mean.
- Maintainer-facing docs explain mock run, real RTX 5090 run, retry, and close
  policy.

## Remaining Real-Hardware Check

The only part that cannot be proven without the RTX 5090 is real scoring
runtime: PyTorch install, CUDA visibility, VRAM measurement, and wall-clock
cost. Everything around that can be tested now with the mock workflow.

# predict-rlm observations (do not fix here)

Notes from exercising Fractal end-to-end on 2026-06-09. Another agent owns
predict-rlm; these are observations only, and the package may have changed
since they were recorded.

- **Sandbox scratch dirs follow the process cwd, not the mounted workspace.**
  Running `fractal --workspace /tmp/fractal-e2e` from the Fractal repo left
  sandbox state under `/Users/emile/git/fractal/.predict_rlm_sbx/...` instead
  of under the target workspace. `sbx ls` shows the sandbox workspace as the
  caller's cwd plus the direct mounts. Harmless for same-directory runs, but
  it litters whatever directory Fractal happens to be launched from and the
  scratch dirs are not cleaned up after `interpreter.shutdown()` (stopped
  sandboxes accumulate in `sbx ls`).

- **Temp-file hook noise on atomic replace.** A single temp-file-plus-
  `os.replace` edit surfaces three separate `os.pwrite` hook events for the
  same `.tmp` file (one per pwrite loop iteration). Fractal renders each as
  "editing .hello.py.tmp". If runtime hooks ever grow coalescing or a
  file-level granularity option, Fractal could drop its display-side
  dedup/compound tracking in `events.py`.

- **`RunTrace.usage` carries no per-iteration cached-token detail at the top
  level.** `prompt_tokens_details.cached_tokens` exists in the raw
  `IterationUsage.main_lm` dicts, but the aggregated `LMUsage`/`TokenUsage`
  does not expose cache reads, so cost-with-cache reporting must re-derive it
  from raw per-step dicts.

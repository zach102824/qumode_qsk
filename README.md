# qumode_qsk

Self-contained quantum-SK Nov3 vs QAOA campaign (variance 1/N, per-instance entropy-peak field).

Phase 1 (freeze, audits, disjoint resource pilot):

    python3 /workspace/qumode_qsk/scripts/run_phase1.py

This does **not** launch the 200-instance main campaign. After the pilot, see `PILOT_RESULTS.json` and `ETA_AND_PROBE.json`.

Science protocol is frozen in `protocol/PROTOCOL_FROZEN.json`. No runtime imports from older scratches. No repo or cloud mutations from this folder.

---

## Published snapshot (2026-11-11 12:09:53 +08 Asia/Shanghai, UTC+8)

This GitHub repository is a **point-in-time** copy of a live CPU campaign that
was still running when the snapshot was taken (`HEARTBEAT.json` event
`scheduler_heartbeat` / `backoff_wait_for_completion`). Results, logs,
references, scans, audits, summaries, starts, and protocol files reflect that
moment. They are not a completed campaign and may be mid-round.

See `PUBLISH_SNAPSHOT.json` for file counts, byte totals, and exclusions.

### How to use this clone (paths are environmental, not physics)

Several scripts, `protocol/PROTOCOL_FROZEN.json` start-file indexes, and many
result JSONs record absolute paths from the original execution root
`/workspace/qumode_qsk`. That is a machine path convention. **The frozen
science protocol was not changed for this publish.**

- Use this repository as the working root (the directory that contains
  `src/`, `protocol/`, and `scripts/`).
- Prefer `python3 scripts/run_phase1.py` from that root rather than the
  hardcoded `/workspace/qumode_qsk/scripts/run_phase1.py` line above.
- Do not expect writes to land under `/workspace/qumode_qsk` unless you
  recreate that layout (or point an environment variable at this clone).

### What was excluded

- Live PID lock: `rounds.lock`
- `__pycache__/`, `*.pyc`, and Numba `*.nbc` / `*.nbi` compiled caches
- No credentials, `.env`, or browser data were present or copied

All campaign data that fit GitHub's limits is included (largest file about
17 MiB N=21 reference NPZ; no file ≥ 100 MiB; no Git LFS; no extra
compression). This snapshot was made without pausing, killing, or
renicing the live campaign, and without modifying the original tree.

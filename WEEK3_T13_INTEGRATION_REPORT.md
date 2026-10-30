# T09/T12/T13 integration status

Branch: feature/t09-t12-t13-integrated
Base: T09/T12 559bdc0; T10/T11 origin/t10_t11 7a8a8bb.

## Integrated behavior
- Preserves deterministic T09 manifest and replay; T10 collector and live display are retained.
- Adds resource-aware CLI with mandatory `--service-profile /path/to/profile.json`.
- Uses T11 `ServiceTimeProfile.from_file` validation.
- Includes running-job estimate, queue service-time sum and per-job estimate for CPU/GPU.
- Rejects unavailable workers; rejects both-ineligible decisions.
- Records scheduler choices as JSONL with both candidate ETAs (unavailable represented as null).

## Verification and limitations
- T09 test subset: 7/7 PASS.
- T13 test subset: 7/7 PASS.
- Full integrated test and end-to-end CUDA still need running in a suitable isolated environment.
- The T11 sample profile from /home/yoo/HDH is not a hardware calibration for Quyets RTX 5060.
- Resource-age fallback currently drops a stale hardware snapshot; a production policy should distinguish stale hardware from fresh worker state.
- Memory admissibility is not yet calibrated: no invented RAM/VRAM thresholds have been added.
- CPU-only currently initializes both backends: benchmarking fairness is separate follow-up work.
- This does not claim T15 benchmark acceptance.

# T09/T12/T13 - Integration and acceptance (2026-10-30)

Branch: `feature/t09-t12-t13-integrated`. Merges T09/T12 commit `559bdc0` and T10/T11 `7a8a8bb`; preserves T09 replay and T10 metrics collector.

## Functional scope
- T09: manifest-based deterministic workload generator and deadline replay.
- T12: Round Robin CPU/GPU assignment.
- T13: Resource-aware scheduler, `--scheduler resource-aware --service-profile <profile.json>`, validated service-time profile, ETA comparison with queued/running work, unavailable worker rejection, per-job `scheduler_decisions.jsonl` (both ETA choices and selected device).
- T10/T11 integrated for live metrics, collection and service-time profiling.

## Verification run on Quyet's WSL Ubuntu-22.04 and RTX 5060 Laptop
- Full regression suite: `PYTHONPATH=src python3 -m unittest discover -s tests -q`: 132 tests, 0 failures, 3 skips, exit code 0.
- Real ONNX smoke using T11 collaborator sample profile: 6 accepted, 6 completed, 0 failed.
- Local T11 hardware calibration: `python3 -m edge_scheduler.inference.profiling --device both --models mobilenetv2 resnet50 --runs 5 --repeats 1 --warmup 1 --no-gpu-telemetry --output-dir /tmp/t13_local_profile`. PASS.
- Local model/device mean times (ms; five samples per cell): MobileNetV2 CPU 18.778, GPU 11.292; ResNet50 CPU 71.134, GPU 15.684. One repeat is a smoke calibration; increase repeats for benchmarking.
- Real ONNX using locally measured profile: `PYTHONPATH=src python3 -m edge_scheduler.main --scheduler resource-aware --service-profile /tmp/t13_local_profile/20261030T140733Z_1163b14e/profile.json --jobs 30 --warmup 1 --no-gpu-telemetry --timeout 90 --output-dir /tmp/t13_e2e_verified`: 30 accepted, 30 completed, 0 failed, 0 unfinished, exit 0.
- 30 decisions persisted to `scheduler_decisions.jsonl`. CPU sessions: CPUExecutionProvider; GPU sessions: CUDAExecutionProvider (with CPU fallback).

## Reproduction
Install dependencies from `requirements.txt`, generate local T11 `profile.json` with above command, then pass its path to `--service-profile`; do not reuse timings from a different machine for benchmark comparisons.

## Remaining out-of-scope / limitations
- This validates T13's functional scheduling and ONNX smoke, not statistical superiority to Round Robin (T15 benchmarking).
- No dedicated calibrated RAM/VRAM per-job admission threshold. Hardware-state freshness fallback uses queue-only ETA when snapshot is stale.
- GPU provider enabled and jobs completed; `--verify-cuda` traces were not collected during these smoke runs.
- 3 tests skipped in full suite; inspect skips before claiming complete coverage.

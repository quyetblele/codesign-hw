# Week 2 - CPU/GPU dispatch and orchestration

## Scope and status
Local worktree: `/home/quyet/t04_github_upload/week2_worktree`, branch `feature/week2-queue-controller`. Original shared branch not modified or pushed.

- Reused `queues.py`, `schedulers.py` and `shutdown.py` from the earlier member-A project folder.
- Added `controller.py`: submission, dispatch, two WorkerLoops, event queue, single-writer Job state updates, error handling, terminal reconciliation, wait and graceful close.
- Original shared contracts remain unchanged: `models.py`, `contracts.py`, `interfaces.py`.
- Added 6 fake-backend integration-style tests and a runnable demo.
- Full unit test suite: **15/15 PASS** (9 previous contract + 6 new).

## Not established
Only FakeBackend is exercised here, NOT real ONNX/CUDA. ResourceMonitor/Collector remain optional dependencies, not production benchmarked. No real inference latency or throughput is claimed.

## Run (Ubuntu/WSL2)
```bash
cd /home/quyet/t04_github_upload/week2_worktree
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/demo_week2.py
git status -sb
```

## Interface handoff
- Backend: `device`, `prepare()`, `infer(model_id, input_ref)`, `close()`.
- Scheduler: `choose_device(job, queued_jobs, snapshot=None)` returns CPU or GPU.
- Controller alone updates Job state; WorkerLoop sends WorkerMessage through a thread-safe queue.
- Each accepted Job must have exactly one terminal result.

## Remaining acceptance gates
- Integrate teammate's real ONNX CPU/GPU backends after reviewing their implementation; no silent CUDA-to-CPU fallback.
- Stress-test shutdown during long inference and submit races, errors from collector, and resource-unavailable behavior.
- Benchmark with real profiles for MobileNetV2/ResNet-50 on each device.
- Document CPU/GPU outputs and end-to-end latency separately.

Python 3.10 references: https://docs.python.org/3.10/library/queue.html ; https://docs.python.org/3.10/library/threading.html

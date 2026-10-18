# WEEK 2 - Member A integration handoff (verified)

## Executive status (18 Oct 2026)
**A-side core READY FOR INTEGRATION WITH B**, not a claim of a real CPU/GPU ONNX demo.

Local WSL worktree: `/home/quyet/t04_github_upload/week2_worktree`
Branch: `feature/week2-queue-controller`

## Project scope and ownership
The six-week group plan assigns Week 2 to Job/Queue/Workers.
A owns Job schema/lifecycle, Incoming/CPU/GPU queues, the Controller, dispatch coordination,
WorkerLoop event routing, basic synchronization and the acceptance tests.
B owns the actual ONNX CPU/GPU InferenceBackend and monitoring/collector implementations.
The original interface contract was preserved; B's code was not edited.

## Deliverables (implemented)
- `src/edge_scheduler/models.py`, `interfaces.py`, `contracts.py`: original shared interface contract kept unchanged.
- `queues.py`: reuse of old JobQueues enhanced with atomic duplicate guarding, scheduled-job guard,
  terminal reconciliation, and `queued_snapshot()` API that encapsulates Queue internals.
- `schedulers.py`: CPU-only, GPU-only, Round Robin and Resource-aware (profile-based) selection.
  The resource-aware profile used in unit tests is FAKE and not a measured CPU/GPU benchmark.
- `shutdown.py`: existing shutdown event utility kept (Controller now uses its own separate acceptance/worker-stop signals).
- `controller.py`: ControllerPort-compliant submit/handle_worker_message/request_shutdown;
  dispatcher; one WorkerLoop per device; thread-safe WorkerMessage queue; Controller-owned
  states and JobEvents; collector error reporting; graceful drain/close and timeout retry.
- `scripts/demo_week2.py`: reproducible 4-Job fake-backend demonstration.
- `tests/test_week2_controller.py`: six initial integration-style tests.
- `tests/test_week2_hardening.py`: 15 additional tests including concurrency, shutdown,
  unavailable worker, slow backend, event errors, duplicate terminal and 300 mixed Jobs.

## Verified tests
Run:
```bash
cd /home/quyet/t04_github_upload/week2_worktree
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/demo_week2.py
```
Last verification: **30/30 PASS for five consecutive complete test runs** (9 contract, 6 controller, 15 hardening).
Stress example: **300 accepted = 300 terminal** (282 completed, 18 deliberately failed).
Round Robin demo: 4 accepted, 4 terminal; 3 completed and 1 deliberately failed.
All timings are tests of fake backend logic; **not** inference performance.

## Design decisions and limitations
1. `request_shutdown()` stops acceptance only; it does not stop running workers.
   `close()` continues dispatching accepted Jobs and draining messages before stopping/joining
   both non-daemon Worker threads and closing backend resources.
2. If `close(timeout=...)` times out, workers are intentionally left running; retry `close()`.
   This prevents loss of accepted Jobs but callers must retry to release resources.
3. Only Controller changes Job states. WorkerLoop reports STARTED/COMPLETED/FAILED.
4. Collector record failures do not lose Jobs: they are gathered in `collector_errors`.
   `close()` still attempts all cleanup then raises to expose logging failure.
5. Queue snapshots are membership snapshots of *queued* jobs; not atomic with running-worker
   resource snapshots. Scheduler remains best effort under concurrency.
6. `Queue` uses Python 3.10 and its protected CPython deque only inside `JobQueues`.
7. Monitor/device unavailable states are rejected as FAILED; automatic retry is outside MVP.
8. Priority field is stored, not yet actively used for scheduling.
9. No formal latency, resource utilization or reproducible real-GPU benchmark is claimed.
10. In-process fake tests do not cover a hard kill of a worker process or a machine power loss.

## Review/checklist before group-wide end-to-end acceptance
- [x] Job/Queue/Controller/WorkerLoop fake-backend flow
- [x] Dual CPU/GPU WorkerLoop with Round Robin test
- [x] Duplicate Job guard and terminal reconciliation
- [x] Shutdown with backlog / in-flight work / acceptance race
- [x] Timeout/retry, late worker message, collector and backend errors
- [x] Controller-side Job schema validation before acceptance and partial prepare cleanup
- [x] Static local commit on separate branch; no push/merge
- [ ] Integrate B's real CPU/GPU ONNX Runtime backends
- [ ] Verify GPU CUDAExecutionProvider truly executes and no unintended CPU fallback
- [ ] Integrate B's ResourceMonitor, MetricsCollector, device profiles
- [ ] End-to-end multi-model benchmark, reporting, review and merge by group

## External technical references
Python 3.10 queue: https://docs.python.org/3.10/library/queue.html
Python 3.10 threading: https://docs.python.org/3.10/library/threading.html

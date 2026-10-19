# T04 – Integration Contract v1 (implementation review)
Status: T04 IMPLEMENTED AS CONTRACT; A/B REVIEW PENDING. Update 2026-10-18:
T06 CPUBackend/GPUBackend are implemented and verified; see T06_report.md.
T08 Resource Monitor is now implemented; see T08_report.md. Controller,
WorkerLoop and MetricsCollector remain outside this implementation.

## 1. Components and ownership
| Component | Owner | Code/contract | Main purpose |
| --- | --- | --- | --- |
| Job, JobStatus, Device | A | models.py | Single shared job data / lifecycle |
| Queue | A / T05 | queues.py | Incoming queue, per-device queues, reconciliation |
| Scheduler | A / T12–T13 | interfaces.py, schedulers.py | choose_device only |
| Controller | A / T07 | ControllerPort in contracts.py | Submit, receive worker messages, initiate shutdown |
| WorkerLoop | A / T07 | WorkerMessage in contracts.py | Consume queued Job; emit events to Controller |
| CPU/GPU Backend | B / T06 | InferenceBackend in contracts.py | prepare, infer, close |
| Monitor | B / T08 | ResourceMonitor in contracts.py | snapshot() to ResourceSnapshot |
| Collector | B / T10 | MetricsCollector in contracts.py | record, flush, close |

## 2. Contract direction
Workload -> JobQueues.accept -> Controller -> Scheduler.choose_device(job, queued_jobs, snapshot) -> JobQueues.dispatch -> device queue -> WorkerLoop -> backend.infer(model_id, input_ref) -> WorkerMessage -> Controller -> Job.mark_running/completed/failed -> JobQueues.record_terminal -> MetricsCollector.record(JobEvent).

The planned accept/dispatch methods update WAITING/SCHEDULED respectively.
queues.py is not present in this checkout. Only Controller should call these
methods and handle state updates after dispatch. The Backend does not mutate Job.

## 3. Frozen names for review
- model_id: mobilenetv2 or resnet50 (not the .onnx filename).
- input_ref: non-empty repository-relative image path. Path traversal restrictions and root resolution belong to T06.
- device: Device.CPU or Device.GPU.
- Scheduler: choose_device(job, queued_jobs, snapshot=None) -> Device.
- InferenceBackend: device property; prepare() -> None; infer(model_id, input_ref) -> numpy.ndarray; close() -> None.
- ResourceMonitor: snapshot() -> ResourceSnapshot.
- T08 implementation additionally exposes start/close, read_sample, diagnostics.
- ResourceSampleSink.record_resource(ResourceSample) is separate from JobEvent logging.
- MetricsCollector: record(JobEvent), flush(), close().
- ControllerPort: submit(Job), handle_worker_message(WorkerMessage), request_shutdown().
- JobEvent: EventKind ACCEPTED / DISPATCHED / STARTED / COMPLETED / FAILED; id, model_id, timestamp_s, optional device/top1/error.
- WorkerMessage: EventKind STARTED / COMPLETED / FAILED; id, device, timestamp, output/error.

## 4. Time and lifecycle rules
- Use monotonic seconds for all execution timestamps on one host.
- Set actual Job arrival_time at acceptance during replay, not at manifest generation.
- Normal: CREATED -> WAITING -> SCHEDULED -> RUNNING -> COMPLETED.
- Failure may lead to FAILED from a nonterminal state.
- COMPLETED/FAILED are terminal and should never be changed again.
- One accepted job_id gets exactly one terminal outcome.
- Worker thread reports messages through a thread-safe event queue; Controller owns Job state after dispatch.
- queued_jobs passed to Scheduler excludes the currently running Job (estimated_remaining_s accounts for it).
- A worker marked unavailable with no runnable capacity must not receive a newly dispatched job; T13 needs a no-capacity policy.
- ResourceSnapshot missing fields use None, never fabricated zeros.
- Backend failure on GPU must not silently fall back to CPU.

The current Job.mark_* methods do NOT enforce all the rules above. T05/T07 must implement the state owner/guards before end-to-end acceptance. This contract does not change old lifecycle methods, avoiding regressions.

## 5. Test command and status
Run:
    .venv/bin/python -m unittest discover -s tests -p test_t04_contract.py -v

Current checkout: 9 T04 tests PASS. The old test_t04_contract_unittest.py and
smoke_member_a.py paths are not present here.
These tests exercise FakeBackend only. They are NOT measurements of CPU/GPU ONNX inference.

## 6. Handoff and unresolved gates
A should build T07 Controller + WorkerLoop using ControllerPort/WorkerMessage.
T06 CPUBackend/GPUBackend now implement InferenceBackend without changing its
signatures. T08 ResourceMonitor and T10 MetricsCollector still use the contracts.
A/B must agree on API before treating v1 as frozen.

T08 now supplies system_probe.py, gpu_probe.py, resource_monitor.py and an
asynchronous resource_log.py sink. T07 passes quick thread-safe worker/queue
state callbacks; monitor never reads backend sessions or changes Job state.
WorkerState appends running_started_s=None and estimated_remaining_known=False,
preserving old positional arguments (covered by a T08 compatibility test).
Idle estimates are known zero; running estimates stay unknown until T11 supplies
them. available means ready/healthy, including a busy worker with a queue.
Missing NVML telemetry is not a backend health failure. Snapshot timestamp is
the conservative hardware sample start; stale policy remains with Controller/T13.

Further items outside T04 contract implementation:
- JobQueues thread-safe queued_jobs snapshot and consumer methods (T05)
- Real ONNX Backend and ResNet50 CUDA verification (T06): implemented; evidence in T06_report.md
- Controller and full event-loop processing (T07)
- Resource availability and all-inf scheduler handling (T13)
- Graceful shutdown and terminal reconciliation under concurrency (T14)
- End-to-end benchmark data (T15)

## 7. Compatibility policy
Do not modify signatures already used in T05/T09/T12/T13 without adding a migration test. Define additions in contracts.py and contract_validation.py. Imported numpy is optional until T06 uses backend output validation; backend runtime is not required for contract tests.

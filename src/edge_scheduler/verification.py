"""Read ORT operator traces as backend evidence, not scheduler profiling T11."""
import json
from collections import Counter
from pathlib import Path


def summarize_profile(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        events = json.load(stream)
    providers = Counter()
    cuda_compute = Counter()
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider")
        if event.get("cat") != "Node" or not provider:
            continue
        providers[provider] += 1
        # Require real model compute, not merely a CUDA memory-copy event.
        op = args.get("op_name", "")
        if provider == "CUDAExecutionProvider" and op in {
            "Conv", "FusedConv", "Gemm", "FusedGemm", "MatMul",
        }:
            cuda_compute[op] += 1
    return {
        "provider_node_events": dict(providers),
        "cuda_compute_events": dict(cuda_compute),
        "cuda_compute_verified": bool(cuda_compute),
    }

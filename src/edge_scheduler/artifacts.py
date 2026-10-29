"""Run directories and discoverable metadata, independent of folder names."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from uuid import uuid4

from .inference.model_registry import PROJECT_ROOT
from .runtime.run_results import write_json


def new_run(output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    directory = output_dir.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def mark_latest(directory: Path, run_type: str, status: str) -> None:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    pointer = {"run_id": directory.name, "path": str(directory.resolve()),
               "run_type": run_type, "status": status,
               "started_at": config.get("started_at", directory.name)}
    targets = [directory.parent / "latest.json"]
    if directory.parent in (PROJECT_ROOT / "logs" / "pipeline", PROJECT_ROOT / "logs" / "profiling"):
        targets.append(PROJECT_ROOT / "logs" / "latest.json")
    for target in targets:
        with target.with_name(".latest.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                previous = json.loads(target.read_text(encoding="utf-8"))
                if previous.get("started_at", "") > pointer["started_at"]:
                    continue
            write_json(target, pointer)

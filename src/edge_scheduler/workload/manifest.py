"""Portable deterministic versioned manifest for workload arrivals."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from hashlib import sha256
import math
import json

MODEL_IDS = frozenset(("mobilenetv2", "resnet50"))
PATTERNS = frozenset(("batch", "low", "medium", "high", "burst"))
MAX_JOBS = 50000

@dataclass(frozen=True, slots=True)
class WorkloadItem:
    job_id: str
    model_id: str
    input_ref: str
    arrival_offset_s: float

    def __post_init__(self):
        if not isinstance(self.job_id, str) or not self.job_id.strip():
            raise ValueError("empty job ID")
        if self.model_id not in MODEL_IDS:
            raise ValueError("unsupported model")
        if not isinstance(self.input_ref, str) or not self.input_ref or "\\" in self.input_ref or "\x00" in self.input_ref:
            raise ValueError("invalid input path")
        path = PurePosixPath(self.input_ref)
        if path.is_absolute() or any(x in (".", "..") for x in self.input_ref.split("/")):
            raise ValueError("unsafe input path")
        t = self.arrival_offset_s
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError("invalid arrival offset")

    def to_dict(self):
        return dict(job_id=self.job_id, model_id=self.model_id, input_ref=self.input_ref,
                    arrival_offset_s=float(self.arrival_offset_s))

@dataclass(frozen=True, slots=True)
class WorkloadManifest:
    pattern: str
    seed: int
    items: tuple[WorkloadItem, ...]
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported schema")
        if self.pattern not in PATTERNS:
            raise ValueError("unknown pattern")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("invalid seed")
        if not isinstance(self.items, tuple) or not 1 <= len(self.items) <= MAX_JOBS:
            raise ValueError("invalid number of jobs")
        seen = set()
        previous = -1.0
        for item in self.items:
            if not isinstance(item, WorkloadItem):
                raise ValueError("invalid item")
            if item.job_id in seen:
                raise ValueError("duplicate ID")
            if item.arrival_offset_s < previous:
                raise ValueError("unsorted offsets")
            previous = item.arrival_offset_s
            seen.add(item.job_id)

    def to_dict(self):
        return dict(schema_version=self.schema_version, pattern=self.pattern,
                    seed=self.seed, items=[x.to_dict() for x in self.items])

    def canonical_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)

    @property
    def sha256(self):
        return sha256(self.canonical_json().encode("utf8")).hexdigest()

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "pattern", "seed", "items"}:
            raise ValueError("invalid manifest schema")
        rows = raw["items"]
        if not isinstance(rows, list) or len(rows) > MAX_JOBS:
            raise ValueError("invalid manifest items")
        items = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"job_id", "model_id", "input_ref", "arrival_offset_s"}:
                raise ValueError("invalid manifest item fields")
            items.append(WorkloadItem(**row))
        return cls(raw["pattern"], raw["seed"], tuple(items), raw["schema_version"])

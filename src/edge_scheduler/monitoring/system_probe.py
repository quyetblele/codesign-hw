"""psutil adapter. start/read/close are called only by the sampler thread."""
from __future__ import annotations

import math
import os
import platform

import psutil

from .monitoring_models import MB, SYSTEM_METRICS, ProbeReading


def error_reason(exc: Exception) -> str:
    if isinstance(exc, (psutil.AccessDenied, PermissionError)):
        return "permission_denied"
    if isinstance(exc, psutil.NoSuchProcess):
        return "process_unavailable"
    return f"read_error:{type(exc).__name__}:{exc}"


class SystemProbe:
    def __init__(self, api=psutil):
        self.api = api
        self._process = None
        self._cpu_primed = False
        self._process_primed = False
        self.metadata = {}

    def start(self) -> None:
        self._cpu_primed = self._process_primed = False
        self.metadata = {
            "platform": platform.platform(), "logical_cpus": self.api.cpu_count(),
            "pid": os.getpid(), "memory_unit": "MB (bytes / 1000000)",
            "scope": "Linux/WSL guest" if "microsoft" in platform.release().lower() else "OS visible resources",
        }
        try:
            self.api.cpu_percent(interval=None)
            self._cpu_primed = True
        except Exception:
            pass
        try:
            self._process = self.api.Process()
            self._process.cpu_percent(interval=None)
            self._process_primed = True
        except Exception:
            self._process = None

    def read(self) -> ProbeReading:
        values = dict.fromkeys(SYSTEM_METRICS)
        errors = {}
        try:
            value = float(self.api.cpu_percent(interval=None))
            if not self._cpu_primed:
                self._cpu_primed = True
                errors["cpu_load_pct"] = "warming_up"
            elif math.isfinite(value) and 0 <= value <= 100:
                values["cpu_load_pct"] = value
            else:
                errors["cpu_load_pct"] = "invalid_value"
        except Exception as exc:
            errors["cpu_load_pct"] = error_reason(exc)
        try:
            memory = self.api.virtual_memory()
            if not (memory.total > 0 and 0 <= memory.available <= memory.total
                    and math.isfinite(memory.percent) and 0 <= memory.percent <= 100):
                raise ValueError("invalid RAM metrics")
            values.update(ram_total_mb=memory.total / MB,
                          ram_free_mb=memory.available / MB, ram_used_pct=float(memory.percent))
        except Exception as exc:
            for key in ("ram_total_mb", "ram_free_mb", "ram_used_pct"):
                errors[key] = error_reason(exc)
        try:
            if self._process is None:
                self._process = self.api.Process()
            rss = self._process.memory_info().rss
            if rss < 0:
                raise ValueError("negative RSS")
            values["process_rss_mb"] = rss / MB
        except Exception as exc:
            errors["process_rss_mb"] = error_reason(exc)
        try:
            if self._process is None:
                self._process = self.api.Process()
            value = float(self._process.cpu_percent(interval=None))
            if not self._process_primed:
                self._process_primed = True
                errors["process_cpu_pct"] = "warming_up"
            elif math.isfinite(value) and value >= 0:
                values["process_cpu_pct"] = value
            else:
                errors["process_cpu_pct"] = "invalid_value"
        except Exception as exc:
            errors["process_cpu_pct"] = error_reason(exc)
        return ProbeReading(values, errors)

    def close(self) -> None:
        self._process = None

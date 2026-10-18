from __future__ import annotations

from dataclasses import dataclass
from threading import Event


@dataclass(slots=True)
class ShutdownController:
    stop_accepting: Event
    force_stop: Event

    @classmethod
    def create(cls) -> "ShutdownController":
        return cls(stop_accepting=Event(), force_stop=Event())

    def request_graceful(self) -> None:
        self.stop_accepting.set()

    def request_force(self) -> None:
        self.stop_accepting.set()
        self.force_stop.set()

    @property
    def accepting_jobs(self) -> bool:
        return not self.stop_accepting.is_set()

    @property
    def forced(self) -> bool:
        return self.force_stop.is_set()

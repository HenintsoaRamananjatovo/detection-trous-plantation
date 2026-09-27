from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .pipeline import run_pipeline
from .types import PipelineConfig


@dataclass
class Job:
    id: str
    state: str = "queued"
    phase: str = "attente"
    progress: float = 0.0
    message: str = "En attente"
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    result: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "phase": self.phase,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, config: PipelineConfig) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=self._execute,
            args=(job.id, config),
            daemon=True,
            name=f"plantation-inference-{job.id[:8]}",
        )
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def has_active_job(self) -> bool:
        with self._lock:
            return any(job.state in {"queued", "running"} for job in self._jobs.values())

    def _execute(self, job_id: str, config: PipelineConfig) -> None:
        self._update(
            job_id,
            state="running",
            phase="initialisation",
            message="Démarrage",
        )

        def progress(phase: str, fraction: float, message: str) -> None:
            self._update(
                job_id,
                phase=phase,
                progress=fraction,
                message=message,
            )

        try:
            result = run_pipeline(config, progress)
            self._update(
                job_id,
                state="completed",
                phase="termine",
                progress=1.0,
                message="Traitement terminé",
                result=result.as_dict(),
            )
        except Exception as error:  # The web API must expose unexpected worker errors.
            self._update(
                job_id,
                state="failed",
                phase="erreur",
                message="Le traitement a échoué",
                error=str(error),
                traceback=traceback.format_exc(),
            )

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)

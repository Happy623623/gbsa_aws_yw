# Created: 2026-08-21 09:29
"""Run several local worker processes and stop them together."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from interview_evidence.runtime.controlproof_model_substitute import (
    controlproof_ai_isolation_digest,
)


@contextmanager
def _exclusive_pool_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("ControlProof worker pool is already active") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("ControlProof worker pool is already active") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_session_manifest(
    root: Path, *, session_id: str, isolation_digest: str, worker_pids: list[int]
) -> Path:
    path = root / "worker-session.json"
    temporary = root / f".worker-session-{os.getpid()}.tmp"
    payload = {
        "schema_version": "controlproof.n02-worker-session.v1",
        "session_id": session_id,
        "launcher_pid": os.getpid(),
        "expected_worker_count": len(worker_pids),
        "worker_pids": worker_pids,
        "ai_isolation_digest": isolation_digest,
        "heartbeat_at": datetime.now(UTC).isoformat(),
    }
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.05)
    return path


def _attested_worker_pids(
    root: Path, *, session_id: str, isolation_digest: str, concurrency: int
) -> list[int] | None:
    directory = root / "worker-attestations"
    if not directory.exists():
        return None
    slots: dict[int, int] = {}
    for path in directory.glob(f"{session_id}-*.json"):
        try:
            proof = json.loads(path.read_text(encoding="utf-8"))
            slot = proof["worker_slot"]
            pid = proof["worker_pid"]
            fresh = (
                datetime.now(UTC) - datetime.fromisoformat(proof["heartbeat_at"])
            ).total_seconds()
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (
            proof.get("session_id") != session_id
            or proof.get("launcher_pid") != os.getpid()
            or proof.get("ai_isolation_digest") != isolation_digest
            or not isinstance(slot, int)
            or not isinstance(pid, int)
            or not 0 <= slot < concurrency
            or not 0 <= fresh < 10
            or path.name != f"{session_id}-{pid}.json"
            or slot in slots
        ):
            return None
        slots[slot] = pid
    if len(slots) != concurrency or len(set(slots.values())) != concurrency:
        return None
    return [slots[slot] for slot in range(concurrency)]


def main() -> None:
    concurrency = int(os.environ.get("WORKER_CONCURRENCY", "4"))
    if concurrency < 1:
        raise SystemExit("WORKER_CONCURRENCY must be at least 1")
    enabled = os.environ.get(
        "CONTROLPROOF_MODEL_SUBSTITUTE_ENABLED", "false"
    ).strip().casefold() in {
        "1", "true", "yes", "on"
    }
    root = Path(os.environ["CONTROLPROOF_OBSERVER_ROOT"]).resolve() if enabled else None
    isolation_digest = controlproof_ai_isolation_digest(os.environ) if enabled else None
    session_id = str(uuid4()) if enabled else None
    lock = _exclusive_pool_lock(root / "worker-pool.lock") if root else nullcontext()
    with lock:
        _run_pool(
            concurrency,
            root=root,
            session_id=session_id,
            isolation_digest=isolation_digest,
        )


def _run_pool(
    concurrency: int, *, root: Path | None, session_id: str | None, isolation_digest: str | None
) -> None:
    child_environment = os.environ.copy()
    if root is not None:
        child_environment["CONTROLPROOF_WORKER_SESSION_ID"] = str(session_id)
        child_environment["CONTROLPROOF_WORKER_LAUNCHER_PID"] = str(os.getpid())
    processes = []
    manifest = None
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for slot in range(concurrency):
            environment = child_environment.copy()
            if root is not None:
                environment["CONTROLPROOF_WORKER_SLOT"] = str(slot)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", "interview_evidence.worker"],
                    env=environment if root is not None else None,
                )
            )
        print(f"Started {concurrency} workers. Press Ctrl+C to stop them.", flush=True)
        while not stopping:
            failed = next((process for process in processes if process.poll() is not None), None)
            if failed is not None:
                raise SystemExit(f"worker exited unexpectedly with code {failed.returncode}")
            if root is not None:
                worker_pids = _attested_worker_pids(
                    root,
                    session_id=str(session_id),
                    isolation_digest=str(isolation_digest),
                    concurrency=concurrency,
                )
                if worker_pids is not None:
                    manifest = _write_session_manifest(
                        root,
                        session_id=str(session_id),
                        isolation_digest=str(isolation_digest),
                        worker_pids=worker_pids,
                    )
                elif manifest is not None:
                    manifest.unlink(missing_ok=True)
                    manifest = None
            time.sleep(0.5)
    finally:
        if manifest is not None:
            for attempt in range(20):
                try:
                    manifest.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if os.name != "nt" or attempt == 19:
                        raise
                    time.sleep(0.05)
        if os.name == "nt":
            # The uv-managed Python launcher can leave its real interpreter behind when only
            # the immediate process is terminated. Kill each explicitly scoped worker tree.
            for process in processes:
                if process.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        capture_output=True,
                    )
        else:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Workers stopped.", flush=True)


if __name__ == "__main__":
    main()

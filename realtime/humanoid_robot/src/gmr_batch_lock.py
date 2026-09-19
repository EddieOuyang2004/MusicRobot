"""Crash-released OS lock, with conservative migration of old PID-only locks."""
from contextlib import contextmanager
import json
import os
from pathlib import Path


class BatchAlreadyRunning(RuntimeError):
    pass


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only.
        if not handle:
            # Access denied or an unexpected error is not proof of a dead owner.
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no PID.
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _lock(handle, acquire):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


def _metadata(handle, status):
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(dict(lock_format=2, pid=os.getpid(), status=status)).encode())
    handle.flush()


@contextmanager
def batch_lock(path: Path):
    # Keep the file permanently: unlinking a locked file creates inode/handle races.
    # Exclusivity belongs to the open OS lock, not to the existence of this file.
    with path.open("a+b") as handle:
        try:
            _lock(handle, True)
        except OSError as exc:
            raise BatchAlreadyRunning(
                f"Another GMR v2 batch is using {path.parent}. Wait for it to finish; do not delete its lock."
            ) from exc
        try:
            handle.seek(0)
            previous = handle.read().decode("utf-8").strip()
            # The predecessor used an exclusive-create PID file and no OS lock.
            # Never take over from a still-running predecessor.
            if previous.isdecimal():
                pid = int(previous)
                if process_alive(pid):
                    raise BatchAlreadyRunning(f"GMR v2 batch PID {pid} is still running; wait for it to finish.")
                print(f"Recovered stale GMR batch lock (PID {pid} no longer exists).", flush=True)
            elif previous:
                try:
                    metadata = json.loads(previous)
                except ValueError as exc:
                    raise BatchAlreadyRunning(f"Unrecognized lock contents in {path}; verify its owner before removing it.") from exc
                if not isinstance(metadata, dict) or metadata.get("lock_format") != 2:
                    raise BatchAlreadyRunning(f"Unrecognized lock format in {path}.")
            _metadata(handle, "running")
            try:
                yield
            finally:
                _metadata(handle, "idle")
        finally:
            _lock(handle, False)

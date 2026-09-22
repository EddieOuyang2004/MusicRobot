"""Bounded scheduling and Windows resource containment for FineDance workers."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import builtins
import ctypes
from ctypes import wintypes as w
import os
import queue
import subprocess
import threading
import time


class BatchStopped(RuntimeError):
    pass


# A blocked console must never block checkpointing or scheduling. Drop old console
# lines when full; progress.json is the authoritative current status.
_console = queue.Queue(maxsize=64)

def _console_writer():
    while True:
        message = _console.get()
        try:
            builtins.print(message, flush=True)
        except (OSError, ValueError):
            pass

threading.Thread(target=_console_writer, daemon=True).start()

def console(*values, **kwargs):
    message = " ".join(map(str, values))
    try:
        _console.put_nowait(message)
    except queue.Full:
        pass


def available_memory_gb():
    if os.name != "nt":
        return float("inf")
    class Memory(ctypes.Structure):
        _fields_ = [("length", w.DWORD), ("load", w.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return value.available / 2**30


class WindowsJob:
    """Kill-on-close job: bounds committed memory/CPU for launcher + descendants."""
    def __init__(self, memory_gb, cpu_percent):
        self.handle = None
        if os.name != "nt":
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                       ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                       ("active", w.DWORD), ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("ro", "wo", "oo", "rb", "wb", "ob")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                       ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        class Cpu(ctypes.Structure):
            _fields_ = [("flags", w.DWORD), ("rate", w.DWORD)]
        limits = Extended()
        limits.basic.flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        limits.job_memory = int(memory_gb * 2**30)
        cpu = Cpu(1 | 4, int(cpu_percent * 100))  # ENABLE | HARD_CAP
        try:
            for kind, value in ((9, limits), (15, cpu)):
                if not kernel.SetInformationJobObject(self.handle, kind, ctypes.byref(value), ctypes.sizeof(value)):
                    raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def assign(self, process):
        if self.handle and not self.kernel.AssignProcessToJobObject(self.handle, w.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_worker(command, *, stdout, env, args):
    stop = getattr(args, "stop_event", None)
    if stop is not None and stop.is_set():
        raise BatchStopped("Batch stopped before worker launch")
    if available_memory_gb() < args.min_free_memory_gb + args.worker_memory_gb:
        if stop is not None:
            stop.set()
        raise BatchStopped("Insufficient free memory to safely launch another worker; resume later")
    job = WindowsJob(args.worker_memory_gb, args.worker_cpu_percent)
    process = None
    assigned = False
    start = time.monotonic()
    try:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, env=env,
                                   creationflags=subprocess.IDLE_PRIORITY_CLASS if os.name == "nt" else 0)
        job.assign(process)
        assigned = True
        while process.poll() is None:
            if stop is not None and stop.wait(.5):
                raise BatchStopped("Batch cancelled; worker terminated")
            if stop is None:
                time.sleep(.5)
            if time.monotonic() - start > args.worker_timeout:
                raise TimeoutError(f"Worker exceeded {args.worker_timeout:g} seconds; resume can retry")
            if available_memory_gb() < args.min_free_memory_gb:
                if stop is not None:
                    stop.set()
                raise BatchStopped("System free memory below reserve; stopping batch")
        return subprocess.CompletedProcess(command, process.returncode)
    finally:
        # Closing the Windows job also kills descendants if the launcher exited.
        if process is not None and process.poll() is None:
            if os.name == "nt" and not assigned:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                process.kill()
        job.close()
        if process is not None:
            process.wait(timeout=10)


def bounded_results(clips, jobs, work, stop, heartbeat):
    """At most jobs futures exist. No new job until the caller handles results."""
    iterator = iter(clips)
    executor = ThreadPoolExecutor(max_workers=jobs)
    pending = {}
    try:
        while True:
            if stop.is_set():
                raise BatchStopped("Batch stopped; completed artifacts remain resumable")
            while len(pending) < jobs:
                clip = next(iterator, None)
                if clip is None:
                    break
                pending[executor.submit(work, clip)] = clip
            if not pending:
                break
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            heartbeat(list(pending.values()))
            for future in done:
                clip = pending.pop(future)
                yield clip, future
    finally:
        stop.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

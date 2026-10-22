"""Pengukuran sumber daya ringan: puncak RAM proses (psutil) dan puncak memori GPU (nvidia-smi)."""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager

import psutil


class PeakRSS:
    """Mengambil sampel RSS proses ini di thread latar belakang dan menyimpan nilai maksimumnya."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self) -> "PeakRSS":
        self.peak = self._proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_gb(self) -> float:
        return self.peak / 2**30


class PeakVRAM:
    """Membaca memori GPU terpakai lewat `nvidia-smi` (seluruh device, MiB) secara berkala
    dan menyimpan nilai maksimumnya."""

    def __init__(self, interval: float = 0.5, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.baseline_mib: int | None = None
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", f"--id={self.gpu_index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._read())
            time.sleep(self.interval)

    def __enter__(self) -> "PeakVRAM":
        self.baseline_mib = self._read()
        self.peak_mib = self.baseline_mib
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


@contextmanager
def timed():
    """Menghasilkan dict yang kunci 'seconds'-nya diisi saat blok selesai."""
    out = {"seconds": None}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - t0


def system_ram() -> dict:
    m = psutil.virtual_memory()
    return {"total_gb": round(m.total / 2**30, 2), "available_gb": round(m.available / 2**30, 2)}

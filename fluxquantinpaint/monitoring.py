# metrics_recorder.py
import time
import threading
from typing import List, Dict, Optional

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False


class MetricsRecorder:
    """
    Records system metrics every `interval` seconds while running.

    Each metric object has the format:
    {
        "timestamp": 1733226072000,  # milliseconds since epoch
        "cpu": 37.5,                 # CPU usage percent (0–100)
        "gpu": 22.1,                 # GPU usage percent (0–100)
        "memory": 61.8               # RAM usage percent (0–100)
    }
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.metrics: List[Dict[str, float]] = []

    def _sample_once(self) -> None:
        ts_ms = int(time.time() * 1000)

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent

        gpu_percent = 0.0
        if _GPU_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    total = 0.0
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        total += float(util.gpu)
                    gpu_percent = total / device_count
            except Exception:
                gpu_percent = 0.0

        sample = {
            "timestamp": ts_ms,
            "cpu": float(cpu_percent),
            "gpu": float(gpu_percent),
            "memory": float(mem_percent),
        }

        with self._lock:
            self.metrics.append(sample)

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def get_metrics(self) -> List[Dict[str, float]]:
        with self._lock:
            return list(self.metrics)

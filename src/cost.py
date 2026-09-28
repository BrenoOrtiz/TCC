"""Medição de custo computacional: tempo, CPU, memória e FLOPs."""

import threading
import time
from dataclasses import dataclass, field

import psutil
import torch
from torch.utils.flop_counter import FlopCounterMode


def sync_device() -> None:
    """Aguarda kernels assíncronos terminarem para que o wall-clock seja real."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def _accelerator_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 2**20
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**20
    return 0.0


@dataclass
class ResourceMonitor:
    """Context manager: wall-clock, tempo de CPU do processo e pico de memória.

    O pico é amostrado por uma thread em `interval` segundos, então picos muito
    curtos podem escapar — suficiente para comparar braços entre si.
    """

    interval: float = 0.2
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_mb: float = 0.0
    peak_accel_mb: float = 0.0
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    def _sample(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            self.peak_rss_mb = max(self.peak_rss_mb, proc.memory_info().rss / 2**20)
            self.peak_accel_mb = max(self.peak_accel_mb, _accelerator_memory_mb())
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        sync_device()
        self._proc = psutil.Process()
        self._cpu0 = sum(self._proc.cpu_times()[:2])
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        sync_device()
        self.wall_s = time.perf_counter() - self._t0
        self.cpu_s = sum(self._proc.cpu_times()[:2]) - self._cpu0
        self._stop.set()
        self._thread.join()


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def torch_forward_flops(module: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> int:
    """FLOPs de um forward (contados pelo FlopCounterMode do PyTorch)."""
    was_training = module.training
    module.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        module(**inputs)
    module.train(was_training)
    return counter.get_total_flops()


def training_flops(forward_flops_per_sample: float, n_samples: int, epochs: int) -> float:
    """Aproximação padrão: backward ≈ 2× forward, logo treino ≈ 3× forward."""
    return 3.0 * forward_flops_per_sample * n_samples * epochs


def llm_generation_flops(
    n_params: float, prompt_tokens: float, output_tokens: float
) -> float:
    """Decoder-only: ≈ 2 · parâmetros FLOPs por token processado (prefill + decode)."""
    return 2.0 * n_params * (prompt_tokens + output_tokens)

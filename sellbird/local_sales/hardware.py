"""Detect RAM / CPU / GPU and recommend a local model size.

Nothing here needs the network. psutil is optional; each probe has a
stdlib fallback. No GPU is assumed.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

# Qwen3 (Apache-2.0) tags as published in the Ollama library, with approximate
# download sizes in GB (4-bit quantised). Sizes are used only for fit checks.
MODEL_LADDER = [
    {"tag": "qwen3:1.7b", "size_gb": 1.4, "params": "1.7B"},
    {"tag": "qwen3:4b", "size_gb": 2.5, "params": "4B"},
    {"tag": "qwen3:8b", "size_gb": 5.2, "params": "8B"},
    {"tag": "qwen3:14b", "size_gb": 9.3, "params": "14B"},
    {"tag": "qwen3:30b", "size_gb": 19.0, "params": "30B (MoE, 3B active)"},
]

PROFILES = {
    "standard": {
        "description": "Ordinary laptop / CPU-only machine",
        "num_ctx": 4096,
        "num_predict": 700,
        "max_cpu_model": "qwen3:4b",
    },
    "high_performance": {
        "description": "Workstation with >=32 GB RAM or a GPU with enough VRAM",
        "num_ctx": 8192,
        "num_predict": 1200,
        "max_cpu_model": "qwen3:14b",
    },
}


@dataclass
class Hardware:
    os: str
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    gpus: list = field(default_factory=list)  # [{"name":..., "vram_gb":...}]
    vram_gb: float = 0.0
    unified_memory: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def _ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return round(int(re.findall(r"\d+", line)[0]) / 1024**2, 1)
    except Exception:
        pass
    if platform.system() == "Darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"]).strip()
        if raw.isdigit():
            return round(int(raw) / 1024**3, 1)
    if platform.system() == "Windows":
        raw = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory"])
        nums = re.findall(r"\d{6,}", raw)
        if nums:
            return round(int(nums[0]) / 1024**3, 1)
    return 0.0


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    if platform.system() == "Darwin":
        raw = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if raw:
            return raw
    return platform.processor() or platform.machine() or "unknown"


def _gpus() -> tuple[list[dict], bool]:
    gpus: list[dict] = []
    unified = False
    if shutil.which("nvidia-smi"):
        raw = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        )
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[1].replace(".", "").isdigit():
                gpus.append({"name": parts[0], "vram_gb": round(float(parts[1]) / 1024, 1)})
    if not gpus and platform.system() == "Darwin" and platform.machine() == "arm64":
        unified = True
        gpus.append({"name": "Apple Silicon (unified memory)", "vram_gb": 0.0})
    if not gpus and platform.system() == "Windows":
        raw = _run(["wmic", "path", "win32_VideoController", "get", "name"])
        for line in raw.splitlines()[1:]:
            if line.strip():
                # VRAM is not reliably reported by wmic; treated as unknown (0).
                gpus.append({"name": line.strip(), "vram_gb": 0.0})
    return gpus, unified


def detect_hardware() -> Hardware:
    gpus, unified = _gpus()
    ram = _ram_gb()
    vram = max((g["vram_gb"] for g in gpus), default=0.0)
    if unified:
        vram = round(ram * 0.65, 1)  # macOS lets the GPU use roughly 2/3 of RAM
    return Hardware(
        os=f"{platform.system()} {platform.release()}",
        cpu_model=_cpu_model(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=ram,
        gpus=gpus,
        vram_gb=vram,
        unified_memory=unified,
    )


def recommend_model(hw: Hardware | None = None) -> dict:
    """Pick the largest Qwen3 tag that fits with ~30% headroom.

    Rules (documented in docs/TECHNICAL_REPORT.md):
      * GPU path: budget = VRAM. CPU path: budget = 50% of RAM (OS + app headroom).
      * A model fits when size_gb * 1.3 <= budget.
      * CPU-only machines are capped (standard: 4B, high-performance RAM: 14B)
        because larger models are too slow for interactive use without a GPU.
    """
    hw = hw or detect_hardware()
    has_gpu = hw.vram_gb >= 4.0
    budget = hw.vram_gb if has_gpu else hw.ram_gb * 0.5
    profile_name = "high_performance" if (hw.vram_gb >= 12 or hw.ram_gb >= 32) else "standard"
    cap_tag = None if has_gpu else PROFILES[profile_name]["max_cpu_model"]
    chosen = MODEL_LADDER[0]
    for m in MODEL_LADDER:
        if m["size_gb"] * 1.3 <= budget:
            chosen = m
        if cap_tag and m["tag"] == cap_tag:
            break
    reason = (
        f"{'GPU VRAM' if has_gpu else '50% of RAM'} budget {budget:.1f} GB; "
        f"largest fitting model with 30% headroom is {chosen['tag']}"
        + ("" if has_gpu else f" (CPU-only cap: {cap_tag})")
    )
    return {
        "hardware": hw.to_dict(),
        "profile": profile_name,
        "profile_settings": PROFILES[profile_name],
        "recommended_model": chosen["tag"],
        "reason": reason,
        "gpu_required": False,
        "ladder": MODEL_LADDER,
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(recommend_model(), indent=2))

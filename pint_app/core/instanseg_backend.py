
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import warnings

# Suppress known, non-fatal warnings from PyTorch/InstanSeg.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.jit\.load.*deprecated.*",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Sparse CSR tensor support is in beta.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Sparse invariant checks are implicitly disabled.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*CUDA is not available or torch_xla is imported.*",
    category=UserWarning,
)

DEFAULT_INSTANSEG_ENV_NAME = "pint-instanseg"

@dataclass
class InstanSegStatus:
    ok: bool
    status: str
    detail: str = ""


def _conda_executable() -> str | None:
    return shutil.which("mamba") or shutil.which("conda")


def _execute(cmd, timeout=1800):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        detail = "\n".join(
            x for x in (result.stdout, result.stderr) if x.strip()
        )
        return InstanSegStatus(
            ok=result.returncode == 0,
            status="Success" if result.returncode == 0 else "Failed",
            detail=detail,
        )
    except Exception as exc:
        return InstanSegStatus(False, "Failed", str(exc))


def check_instanseg_backend(
    env_name: str = DEFAULT_INSTANSEG_ENV_NAME,
):
    conda = _conda_executable()
    if not conda:
        return InstanSegStatus(
            False, "Conda not found",
            "Conda or Mamba must be available on PATH."
        )

    result = _execute([
        conda, "run", "-n", env_name,
        "python", "-c",
        (
            "import torch, instanseg; "
            "from instanseg import InstanSeg; "
            "print('INSTANSEG_OK'); "
            "print('PyTorch:', torch.__version__); "
            "print('CUDA:', torch.cuda.is_available()); "
            "print('MPS:', torch.backends.mps.is_available())"
        ),
    ], timeout=120)

    return InstanSegStatus(
        result.ok and "INSTANSEG_OK" in result.detail,
        "Available" if result.ok and "INSTANSEG_OK" in result.detail
        else "Not available",
        result.detail,
    )


def install_instanseg_backend(
    env_name: str = DEFAULT_INSTANSEG_ENV_NAME,
):
    conda = _conda_executable()
    if not conda:
        return InstanSegStatus(
            False, "Conda not found",
            "Conda or Mamba must be available on PATH."
        )

    # Reuse a working environment.
    existing = check_instanseg_backend(env_name)
    if existing.ok:
        return existing

    commands = [
        [
            conda, "create", "-n", env_name,
            "python=3.11", "-y"
        ],
        [
            conda, "run", "-n", env_name,
            "python", "-m", "pip", "install",
            "instanseg-torch", "tifffile"
        ],
    ]

    logs = []
    for cmd in commands:
        result = _execute(cmd)
        logs.append(result.detail)
        if not result.ok:
            return InstanSegStatus(
                False, "Installation failed",
                "\n\n".join(logs),
            )

    return check_instanseg_backend(env_name)


def run_instanseg_backend(
    input_path: str | Path,
    out_nuclei_path: str | Path,
    out_cells_path: str | Path,
    out_json_path: str | Path,
    pixel_size: float = 1.0,
    device: str = "auto",
    tile_size: int = 512,
    batch_size: int = 1,
    env_name: str = DEFAULT_INSTANSEG_ENV_NAME,
):
    conda = _conda_executable()
    if not conda:
        return InstanSegStatus(False, "Conda not found")

    runner = Path(__file__).with_name(
        "run_instanseg_backend.py"
    )

    cmd = [
        conda, "run", "-n", env_name,
        "python", str(runner),
        "--input", str(input_path),
        "--out-nuclei", str(out_nuclei_path),
        "--out-cells", str(out_cells_path),
        "--out-json", str(out_json_path),
        "--pixel-size", str(pixel_size),
        "--device", device,
        "--tile-size", str(tile_size),
        "--batch-size", str(batch_size),
    ]

    result = _execute(cmd, timeout=3600)

    if not result.ok:
        return InstanSegStatus(
            False, "Segmentation failed", result.detail
        )

    try:
        summary = json.loads(
            Path(out_json_path).read_text(encoding="utf-8")
        )
        n_cells = summary["n_cells"]
        n_nuclei = summary["n_nuclei"]
    except Exception as exc:
        return InstanSegStatus(
            False, "Output validation failed", str(exc)
        )

    return InstanSegStatus(
        True,
        f"Completed: {n_cells:,} cells, {n_nuclei:,} nuclei",
        result.detail,
    )



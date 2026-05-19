from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


DEFAULT_MESMER_ENV_NAME = "pint-mesmer"


@dataclass
class MesmerStatus:
    ok: bool
    status: str
    detail: str
    conda_executable: str | None = None
    env_name: str = DEFAULT_MESMER_ENV_NAME


def find_conda_executable() -> str | None:
    """
    Find conda or mamba on PATH.

    Returns
    -------
    str | None
        Path/name of conda-like executable, or None if unavailable.
    """
    for exe in ("mamba", "conda"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def get_mesmer_install_commands(env_name: str = DEFAULT_MESMER_ENV_NAME) -> list[str]:
    """
    Commands the user can run manually in a terminal.
    """
    return [
        f"conda create -n {env_name} python=3.10 -y",
        f"conda run -n {env_name} python -m pip install --upgrade pip",
        f"conda run -n {env_name} python -m pip install deepcell tifffile numpy",
    ]


def check_mesmer_backend(env_name: str = DEFAULT_MESMER_ENV_NAME) -> MesmerStatus:
    """
    Check whether the optional Mesmer/DeepCell backend is available
    in a separate conda/mamba environment.

    This function must never import deepcell in the main PINT process.
    """
    conda_exe = find_conda_executable()

    if conda_exe is None:
        return MesmerStatus(
            ok=False,
            status="Conda/mamba not found",
            detail=(
                "Could not find 'conda' or 'mamba' on PATH. "
                "Install Mesmer manually in a separate environment, or start PINT "
                "from a terminal where conda is available."
            ),
            conda_executable=None,
            env_name=env_name,
        )

    cmd = [
        conda_exe,
        "run",
        "-n",
        env_name,
        "python",
        "-c",
        (
            "import deepcell; "
            "from deepcell.applications import Mesmer; "
            "print('MESMER_OK')"
        ),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MesmerStatus(
            ok=False,
            status="Check timed out",
            detail=(
                f"The Mesmer backend check timed out while testing environment '{env_name}'. "
                "The environment may be broken or TensorFlow/DeepCell may be hanging during import."
            ),
            conda_executable=conda_exe,
            env_name=env_name,
        )
    except Exception as e:
        return MesmerStatus(
            ok=False,
            status="Check failed",
            detail=str(e),
            conda_executable=conda_exe,
            env_name=env_name,
        )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if completed.returncode == 0 and "MESMER_OK" in stdout:
        return MesmerStatus(
            ok=True,
            status="Available",
            detail=(
                f"Mesmer/DeepCell is available in environment '{env_name}'.\n\n"
                f"Command used:\n{' '.join(cmd)}"
            ),
            conda_executable=conda_exe,
            env_name=env_name,
        )

    return MesmerStatus(
        ok=False,
        status="Not available",
        detail=(
            f"Could not import Mesmer from environment '{env_name}'.\n\n"
            f"Command used:\n{' '.join(cmd)}\n\n"
            f"stdout:\n{stdout or '—'}\n\n"
            f"stderr:\n{stderr or '—'}"
        ),
        conda_executable=conda_exe,
        env_name=env_name,
    )


def install_mesmer_backend(env_name: str = DEFAULT_MESMER_ENV_NAME) -> MesmerStatus:
    """
    Install Mesmer/DeepCell into a separate conda/mamba environment.

    This is intentionally conservative:
    - it does not modify the main PINT environment
    - it uses subprocess
    - it returns a status object instead of raising where possible
    """
    conda_exe = find_conda_executable()

    if conda_exe is None:
        return MesmerStatus(
            ok=False,
            status="Conda/mamba not found",
            detail=(
                "Could not find 'conda' or 'mamba' on PATH. "
                "Please install manually using the commands shown in the Segmentation tab."
            ),
            conda_executable=None,
            env_name=env_name,
        )

    commands = [
        [conda_exe, "create", "-n", env_name, "python=3.10", "-y"],
        [conda_exe, "run", "-n", env_name, "python", "-m", "pip", "install", "--upgrade", "pip"],
        [conda_exe, "run", "-n", env_name, "python", "-m", "pip", "install", "deepcell", "tifffile", "numpy"],
    ]

    logs: list[str] = []

    for cmd in commands:
        logs.append(f"$ {' '.join(cmd)}")

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return MesmerStatus(
                ok=False,
                status="Installation timed out",
                detail="\n\n".join(logs),
                conda_executable=conda_exe,
                env_name=env_name,
            )
        except Exception as e:
            logs.append(str(e))
            return MesmerStatus(
                ok=False,
                status="Installation failed",
                detail="\n\n".join(logs),
                conda_executable=conda_exe,
                env_name=env_name,
            )

        if completed.stdout.strip():
            logs.append(completed.stdout.strip())

        if completed.stderr.strip():
            logs.append("stderr:")
            logs.append(completed.stderr.strip())

        if completed.returncode != 0:
            return MesmerStatus(
                ok=False,
                status="Installation failed",
                detail="\n\n".join(logs),
                conda_executable=conda_exe,
                env_name=env_name,
            )

    # Re-check after installation.
    status = check_mesmer_backend(env_name=env_name)

    if status.ok:
        status.detail = "Installation completed successfully.\n\n" + status.detail
    else:
        status.detail = (
            "Installation commands completed, but Mesmer could not be imported afterwards.\n\n"
            + status.detail
        )

    return status
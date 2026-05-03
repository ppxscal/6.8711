"""Runtime helpers for subprocesses, environments, and GPU visibility."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    stream: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    if not quiet:
        print("$", " ".join(cmd))
    if stream:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        collected: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            if not quiet:
                print(line, end="")
            collected.append(line)
        returncode = process.wait()
        stdout = "".join(collected)
        completed = subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    else:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
        )
        if not quiet:
            if completed.stdout:
                print(completed.stdout[-4000:])
            if completed.stderr and completed.returncode != 0:
                print(completed.stderr[-4000:])
    if completed.returncode != 0 and check:
        stderr_tail = (completed.stderr or "")[-4000:]
        stdout_tail = (completed.stdout or "")[-4000:]
        raise RuntimeError(
            f"Command failed with code {completed.returncode}: {' '.join(cmd)}\n"
            f"{stderr_tail or stdout_tail}"
        )
    return completed


def env_root(env_name: str, paths) -> Path:
    uv_root = paths.envs_dir / "uv" / env_name
    if uv_root.exists():
        return uv_root
    mamba_root = paths.envs_dir / "micromamba-root" / "envs" / env_name
    if mamba_root.exists():
        return mamba_root
    return paths.envs_dir / env_name


def env_python(env_name: str, paths) -> Path:
    return env_root(env_name, paths) / "bin" / "python"


def env_binary(env_name: str, binary: str, paths) -> Path:
    return env_root(env_name, paths) / "bin" / binary


def discover_devices() -> list[str]:
    requested = os.environ.get("GPU_DEVICES", "").strip()
    if requested:
        devices = [f"cuda:{token.strip()}" for token in requested.split(",") if token.strip()]
        return devices or ["cpu"]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"-1", "NoDevFiles"}:
        devices = [f"cuda:{token.strip()}" for token in visible.split(",") if token.strip()]
        max_gpus = os.environ.get("MAX_GPUS", "").strip()
        if max_gpus:
            try:
                devices = devices[:max(1, int(max_gpus))]
            except ValueError:
                pass
        return devices or ["cpu"]

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        devices = [f"cuda:{line.strip()}" for line in output.splitlines() if line.strip()]
        max_gpus = os.environ.get("MAX_GPUS", "").strip()
        if max_gpus:
            try:
                devices = devices[:max(1, int(max_gpus))]
            except ValueError:
                pass
        if devices:
            return devices
    except Exception:
        pass

    return ["cpu"]


def visible_gpu_env(device: str) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    local_device = device
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = index
        local_device = "cuda:0"
    return env, local_device

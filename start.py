"""同时启动本项目的 FastAPI 后端和 Vite 前端。

可在普通 PowerShell 中执行：``python start.py``，也可双击运行。
本脚本直接调用 node.exe 和 Vite 的 JavaScript 入口，避免经由 npm.cmd。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIRECTORY = PROJECT_ROOT / "backend"
FRONTEND_DIRECTORY = PROJECT_ROOT / "frontend"
BACKEND_PYTHON = BACKEND_DIRECTORY / ".venv" / "Scripts" / "python.exe"
VITE_ENTRY = FRONTEND_DIRECTORY / "node_modules" / "vite" / "bin" / "vite.js"
# Keep the launcher, Vite proxy and browser URL on one port pair.  8101 is
# frequently held by an unrelated stale process on this workstation, which
# previously led to a healthy backend on 8102 but a frontend proxy still
# sending every API request to dead port 8101.
BACKEND_PORT = 8103
FRONTEND_PORT = 5275


def find_node() -> str | None:
    """Find node.exe without going through the npm.cmd wrapper."""
    configured = os.environ.get("NODE_EXE")
    candidates = [configured] if configured else []
    candidates.extend(
        [
            shutil.which("node.exe"),
            shutil.which("node"),
            str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "node.exe"),
        ]
    )
    return next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)


def require_file(path: Path, install_hint: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"未找到 {path.relative_to(PROJECT_ROOT)}。\n请先执行：{install_hint}")


def stop(process: subprocess.Popen[object]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def wait_for_backend(process: subprocess.Popen[object], timeout_seconds: float = 30.0) -> bool:
    """Wait until FastAPI accepts local health checks before starting Vite."""
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://127.0.0.1:{BACKEND_PORT}/api/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (OSError, URLError):
            pass
        time.sleep(0.25)
    return False


def main() -> int:
    try:
        require_file(BACKEND_PYTHON, "cd backend; python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install -r requirements.txt")
        require_file(VITE_ENTRY, "cd frontend; npm install")
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    node = find_node()
    if node is None:
        print("未找到 node.exe。请安装 Node.js，或设置 NODE_EXE 为 node.exe 的完整路径。", file=sys.stderr)
        return 1

    print("正在启动服务（按 Ctrl+C 同时停止）…")
    backend = subprocess.Popen(
        [str(BACKEND_PYTHON), "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
        cwd=BACKEND_DIRECTORY,
    )
    if not wait_for_backend(backend):
        print("后端在 30 秒内未完成启动，未启动前端。请查看上方后端日志。", file=sys.stderr)
        stop(backend)
        return 1
    frontend = subprocess.Popen(
        [node, str(VITE_ENTRY), "--host", "127.0.0.1", "--port", str(FRONTEND_PORT), "--strictPort"],
        cwd=FRONTEND_DIRECTORY,
    )
    print(f"前端：http://127.0.0.1:{FRONTEND_PORT}")
    print(f"后端：http://127.0.0.1:{BACKEND_PORT}/docs")

    processes = (backend, frontend)
    try:
        while True:
            for process in processes:
                if (exit_code := process.poll()) is not None:
                    print(f"一个服务已退出（退出码 {exit_code}），正在停止其余服务。", file=sys.stderr)
                    return exit_code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在停止服务…")
        return 0
    finally:
        for process in processes:
            stop(process)


if __name__ == "__main__":
    raise SystemExit(main())

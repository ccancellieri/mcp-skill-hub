"""SearXNG Docker container lifecycle."""
from __future__ import annotations

from pathlib import Path

from . import _proc
from .base import Service, Status, install_log_path

DOCKER_DESKTOP_URL = "https://www.docker.com/products/docker-desktop"


def _docker_available() -> tuple[bool, str]:
    if _proc.which("docker") is None:
        return False, f"docker not installed — see {DOCKER_DESKTOP_URL}"
    code, out = _proc.run(["docker", "info"], timeout=5)
    if code != 0:
        return False, "docker daemon not running — open Docker Desktop"
    return True, ""


def _container_exists(name: str) -> bool:
    code, out = _proc.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}", "--filter", f"name=^{name}$"],
        timeout=5,
    )
    return code == 0 and name in out.splitlines()


def _container_running(name: str) -> bool:
    code, out = _proc.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        timeout=5,
    )
    return code == 0 and out.strip().lower() == "true"


def _compose_file() -> Path:
    """Path to the shipped docker-compose.searxng.yml."""
    pkg = Path(__file__).resolve().parents[3]  # src/skill_hub/services/ -> repo root
    return pkg / "docker" / "docker-compose.searxng.yml"


class SearxngContainer(Service):
    name = "searxng"
    label = "SearXNG"
    description = "Web-search backend (Docker container)."

    def __init__(self, container: str = "skill-hub-searxng"):
        self.container = container

    def status(self) -> Status:
        ok, _ = _docker_available()
        if not ok:
            return "unavailable"
        if not _container_exists(self.container):
            return "unavailable"
        return "running" if _container_running(self.container) else "stopped"

    def is_available(self) -> tuple[bool, str]:
        ok, msg = _docker_available()
        if not ok:
            return False, msg
        if not _container_exists(self.container):
            return False, "container not created — click Install to run docker-compose"
        return True, ""

    def start(self) -> tuple[bool, str]:
        ok, msg = _docker_available()
        if not ok:
            return False, msg
        if not _container_exists(self.container):
            return False, "container missing — click Install first"
        code, out = _proc.run(["docker", "start", self.container], timeout=15)
        return code == 0, "started" if code == 0 else out.strip()[:200]

    def stop(self) -> tuple[bool, str]:
        if not _container_exists(self.container):
            return True, "no container — already stopped"
        code, out = _proc.run(["docker", "stop", "-t", "5", self.container], timeout=20)
        return code == 0, "stopped" if code == 0 else out.strip()[:200]

    def installable(self) -> bool:
        return _proc.which("docker") is not None

    def install(self) -> tuple[bool, str] | None:
        ok, msg = _docker_available()
        if not ok:
            return False, msg
        compose = _compose_file()
        if not compose.exists():
            return False, f"compose file missing at {compose}"
        return _proc.stream_install(
            ["docker", "compose", "-f", str(compose), "up", "-d", "--pull", "always"],
            install_log_path(self.name),
            timeout=300,
        )

    def resource_footprint(self) -> dict:
        return {"ram_mb_approx": 400, "cpu_share": 0.05}

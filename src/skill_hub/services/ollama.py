"""Ollama daemon and model lifecycle services."""
from __future__ import annotations

from . import _proc
from .base import Service, Status, install_log_path


def _ollama_available() -> tuple[bool, str]:
    if _proc.which("ollama") is None:
        return False, "ollama binary not in PATH — install with `brew install ollama`"
    return True, ""


def _brew_services_manages_ollama() -> bool:
    """True if Homebrew's services manifest lists ollama (preferred on macOS)."""
    if _proc.which("brew") is None:
        return False
    code, out = _proc.run(["brew", "services", "list"], timeout=5)
    if code != 0:
        return False
    for line in out.splitlines():
        if line.split() and line.split()[0] == "ollama":
            return True
    return False


class OllamaDaemon(Service):
    name = "ollama_daemon"
    label = "Ollama daemon"
    description = "Local LLM host. Required by router/embed models."

    def status(self) -> Status:
        ok, _ = _ollama_available()
        if not ok:
            return "unavailable"
        code, _ = _proc.run(["pgrep", "-f", "ollama serve"], timeout=3)
        return "running" if code == 0 else "stopped"

    def is_available(self) -> tuple[bool, str]:
        return _ollama_available()

    def start(self) -> tuple[bool, str]:
        ok, msg = _ollama_available()
        if not ok:
            return False, msg
        if _brew_services_manages_ollama():
            code, out = _proc.run(["brew", "services", "start", "ollama"], timeout=15)
            return code == 0, out.strip() or ("started" if code == 0 else "brew failed")
        return _proc.popen_detached(["ollama", "serve"])

    def stop(self) -> tuple[bool, str]:
        if _brew_services_manages_ollama():
            code, out = _proc.run(["brew", "services", "stop", "ollama"], timeout=15)
            return code == 0, out.strip() or ("stopped" if code == 0 else "brew failed")
        code, out = _proc.run(["pkill", "-TERM", "-f", "ollama serve"], timeout=5)
        return code in (0, 1), "stopped" if code in (0, 1) else out.strip()

    def installable(self) -> bool:
        return _proc.which("brew") is not None

    def install(self) -> tuple[bool, str] | None:
        if _proc.which("brew") is None:
            return None
        return _proc.stream_install(
            ["brew", "install", "ollama"],
            install_log_path(self.name),
            timeout=600,
        )

    def resource_footprint(self) -> dict:
        return {"ram_mb_approx": 150, "cpu_share": 0.02}


def _ollama_ps_has(model: str) -> bool:
    code, out = _proc.run(["ollama", "ps"], timeout=5)
    if code != 0:
        return False
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0].startswith(model):
            return True
    return False


def _ollama_list_has(model: str) -> bool:
    code, out = _proc.run(["ollama", "list"], timeout=5)
    if code != 0:
        return False
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0].startswith(model):
            return True
    return False


class OllamaModel(Service):
    """One Ollama model loaded/unloaded in VRAM.

    Two instances in the default registry: ``ollama_router`` (qwen2.5:3b) and
    ``ollama_embed`` (nomic-embed-text).
    """

    approx_ram_mb: int = 2000

    def __init__(self, name: str, label: str, description: str, model: str, approx_ram_mb: int = 2000):
        self.name = name
        self.label = label
        self.description = description
        self.model = model
        self.approx_ram_mb = approx_ram_mb

    def status(self) -> Status:
        ok, _ = _ollama_available()
        if not ok:
            return "unavailable"
        code, _ = _proc.run(["pgrep", "-f", "ollama serve"], timeout=3)
        if code != 0:
            return "unavailable"
        return "running" if _ollama_ps_has(self.model) else "stopped"

    def is_available(self) -> tuple[bool, str]:
        ok, msg = _ollama_available()
        if not ok:
            return False, msg
        if not _ollama_list_has(self.model):
            return False, f"model not pulled — run `ollama pull {self.model}` or click Install"
        return True, ""

    def start(self) -> tuple[bool, str]:
        if not _ollama_list_has(self.model):
            return False, f"model {self.model} not installed"
        code, out = _proc.run(["ollama", "run", self.model, ""], timeout=30)
        return code == 0, "loaded" if code == 0 else out.strip()[:200]

    def stop(self) -> tuple[bool, str]:
        code, out = _proc.run(["ollama", "stop", self.model], timeout=10)
        return code == 0, "unloaded" if code == 0 else out.strip()[:200]

    def installable(self) -> bool:
        return _proc.which("ollama") is not None

    def install(self) -> tuple[bool, str] | None:
        if _proc.which("ollama") is None:
            return None
        return _proc.stream_install(
            ["ollama", "pull", self.model],
            install_log_path(self.name),
            timeout=600,
        )

    def resource_footprint(self) -> dict:
        return {"ram_mb_approx": self.approx_ram_mb, "cpu_share": 0.1}

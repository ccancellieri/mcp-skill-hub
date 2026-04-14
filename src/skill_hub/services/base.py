"""Service ABC — uniform lifecycle interface for the control panel."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

Status = Literal["running", "stopped", "unavailable", "transitioning"]

INSTALL_LOG_DIR = Path("/tmp")


def install_log_path(service_name: str) -> Path:
    return INSTALL_LOG_DIR / f"skill-hub-install-{service_name}.log"


class Service(ABC):
    """Uniform lifecycle for a managed resource (process, container, task)."""

    #: Stable identifier used in URLs and config keys.
    name: str = ""

    #: Human label for the UI.
    label: str = ""

    #: One-line description shown on the card.
    description: str = ""

    @abstractmethod
    def status(self) -> Status:
        """Return current runtime state. Never raise."""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """
        Return (True, "") when prerequisites are satisfied, else
        (False, "short hint explaining what is missing").
        """

    @abstractmethod
    def start(self) -> tuple[bool, str]:
        """Start the service. Return (ok, message). Never raise."""

    @abstractmethod
    def stop(self) -> tuple[bool, str]:
        """Stop the service. Return (ok, message). Never raise."""

    def installable(self) -> bool:
        """Whether an ``install()`` path exists — UI shows Install button if True."""
        return False

    def install(self) -> tuple[bool, str] | None:
        """
        Attempt to install a missing prerequisite.

        Returns:
            None if no installer exists for this service (UI hides the button).
            Otherwise (ok, message) after the install subprocess completes.
        """
        return None

    def resource_footprint(self) -> dict:
        """
        Best-effort estimate of the service's runtime cost.

        Used by the pressure monitor to decide suggestion priority. Return
        keys: ``ram_mb_approx`` (int), ``cpu_share`` (0..1). Defaults to
        conservative zero.
        """
        return {"ram_mb_approx": 0, "cpu_share": 0.0}

"""Banner middleware — injects disabled_services into request.state on every request.

Lets base.html render the sticky warning banner without every route having to
pass the list into its template context.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class BannerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            from ...services.registry import get_registry

            request.state.disabled_services = get_registry().disabled_services
        except Exception:  # noqa: BLE001
            request.state.disabled_services = []

        # Issue #6 — surface no_llm_mode as a sticky banner so the user knows
        # exactly why LLM-dependent tools are returning the "disabled" message.
        try:
            from ...capabilities import no_llm_mode_active, no_llm_summary

            if no_llm_mode_active():
                ns = no_llm_summary()
                request.state.no_llm_mode = True
                request.state.no_llm_available = ns["available"]
                request.state.no_llm_total = ns["total"]
            else:
                request.state.no_llm_mode = False
        except Exception:  # noqa: BLE001
            request.state.no_llm_mode = False
        return await call_next(request)

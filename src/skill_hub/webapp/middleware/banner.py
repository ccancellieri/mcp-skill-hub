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
        return await call_next(request)

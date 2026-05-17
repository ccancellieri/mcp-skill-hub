"""Capabilities view — /status/capabilities.

Issue #13 (M1): "Is skill-hub useless without a local LLM?" — answer with
a URL. Lists every MCP tool with a green / yellow / red verdict driven by
the current backend probe.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ...capabilities import render_matrix

router = APIRouter()


@router.get("/status/capabilities", response_class=HTMLResponse)
def capabilities_page(request: Request):
    data = render_matrix()
    return request.app.state.templates.TemplateResponse(
        request,
        "capabilities.html",
        {
            **data,
            "active_tab": "capabilities",
        },
    )


@router.get("/api/capabilities")
def capabilities_json() -> JSONResponse:
    return JSONResponse(render_matrix())

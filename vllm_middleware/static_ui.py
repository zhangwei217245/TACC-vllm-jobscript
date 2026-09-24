"""Serve the kit's UI through vLLM: --middleware static_ui.StaticUIMiddleware.

TACC_UI_DIR: static directory; TACC_UI_PAGE: entry file (default chat.html).
TACC_PUBLIC_BASE_URL: optional browser-facing server root, without /v1.
TACC_UI_MODEL: optional default model name. These public settings are returned
by /ui/config.json; no API key or arbitrary environment values are exposed.
With no public URL, the browser uses the origin and prefix of the UI page.
Only /ui and /ui/... HTTP requests are intercepted. All other requests,
WebSockets and lifespan events pass to vLLM unchanged. No SPA fallback.
Legacy VLLM_* inputs remain readable for older launchers; TACC_* takes precedence.
"""

import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

from starlette._utils import get_route_path
from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles


def _setting(suffix: str, default: str = "") -> str:
    # New launcher exports only TACC_*; the fallback supports older launchers.
    return os.environ.get("TACC_" + suffix, os.environ.get("VLLM_" + suffix, default))


class StaticUIMiddleware:
    def __init__(self, app):
        self.app = app
        directory = _setting("UI_DIR")
        if not directory:
            raise RuntimeError("Set TACC_UI_DIR to the frontend directory")
        directory = Path(directory).resolve()
        page = _setting("UI_PAGE", "chat.html")
        relative = PurePosixPath(page)
        if (not page or relative.is_absolute() or ".." in relative.parts
                or any(c in page for c in "\\?#")
                or any(ord(c) < 32 for c in page)
                or relative.suffix.lower() not in {".html", ".htm"}):
            raise RuntimeError("TACC_UI_PAGE must be a relative HTML path inside TACC_UI_DIR")
        entry = (directory / page).resolve()
        if not entry.is_relative_to(directory) or not entry.is_file():
            raise RuntimeError(f"UI entry missing or outside TACC_UI_DIR: {entry}")
        if not os.access(entry, os.R_OK):
            raise RuntimeError(f"UI entry is not readable: {entry}")
        self.page = relative.as_posix()
        self.public_base = _setting("PUBLIC_BASE_URL").strip().rstrip("/")
        if self.public_base:
            url = urlsplit(self.public_base)
            # Public configuration cannot contain credentials, query tokens or fragments.
            if (url.scheme not in {"http", "https"} or not url.hostname
                    or url.username is not None or url.password is not None
                    or url.query or url.fragment
                    or any(c.isspace() or ord(c) < 32 for c in self.public_base)
                    or "\\" in self.public_base):
                raise RuntimeError("TACC_PUBLIC_BASE_URL must be an HTTP(S) server root without credentials/query/fragment")
            _ = url.port  # Validate numeric port syntax/range.
            if url.path.rstrip("/").endswith("/v1"):
                raise RuntimeError("TACC_PUBLIC_BASE_URL is the server root; omit the trailing /v1")
        self.model = _setting("UI_MODEL")
        self.ui = Starlette(routes=[
            Route("/ui/config.json", self.config, methods=["GET", "HEAD"]),
            Route("/ui/", self.landing, methods=["GET", "HEAD"]),
            Mount("/ui", app=StaticFiles(directory=str(directory), html=True, follow_symlink=False)),
        ])

    async def landing(self, request):
        prefix = request.scope.get("root_path", "").rstrip("/")
        return RedirectResponse(prefix + "/ui/" + quote(self.page, safe="/"))

    async def config(self, request):
        # Null means derive from the browser URL, including any gateway prefix.
        # Do not derive a public host from the server's 0.0.0.0 bind address.
        return JSONResponse({
            "api_base_url": self.public_base + "/v1" if self.public_base else None,
            "model": self.model,
        }, headers={"Cache-Control": "no-store"})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = get_route_path(scope)
            if path == "/ui" or path.startswith("/ui/"):
                await self.ui(scope, receive, send)
                return
        await self.app(scope, receive, send)

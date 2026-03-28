from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import parse
from urllib import request


@dataclass
class MyEnvClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 15.0
    session_id: str | None = None

    def reset(self, difficulty: str = "medium", seed: int = 42, session_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"difficulty": difficulty, "seed": seed}
        sid = session_id if session_id is not None else self.session_id
        if sid is not None:
            payload["session_id"] = sid

        response = self._post("/reset", payload)
        self.session_id = str(response.get("session_id")) if response.get("session_id") is not None else self.session_id
        return response

    def step(self, action: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        sid = session_id if session_id is not None else self.session_id
        if sid is None:
            raise RuntimeError("session_id is required. Call reset() first or provide session_id.")
        return self._post("/step", {"session_id": sid, "action": action})

    def state(self, session_id: str | None = None) -> dict[str, Any]:
        sid = session_id if session_id is not None else self.session_id
        if sid is None:
            raise RuntimeError("session_id is required. Call reset() first or provide session_id.")
        query = parse.urlencode({"session_id": sid})
        return self._get(f"/state?{query}")

    def tasks(self) -> dict[str, Any]:
        return self._get("/tasks")

    def grader(self) -> dict[str, Any]:
        return self._get("/grader")

    def baseline(self) -> dict[str, Any]:
        return self._get("/baseline")

    async def websocket_not_implemented(self, _path: str = "/ws") -> None:
        raise NotImplementedError("WebSocket transport is not currently exposed by the server")

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        req = request.Request(url=url, method="GET")
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            url=url,
            method="POST",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            content = response.read().decode("utf-8")
        return json.loads(content)

import asyncio
import json
import subprocess
import uuid
from typing import Any


class EAMcpClient:
    """EA MCP stdio サブプロセスを保持し JSON-RPC を送受信する"""

    def __init__(self, command: list[str]):
        self.command = command
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self.tools: list[dict] = []

    async def start(self):
        kwargs: dict = dict(
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Windows: コンソールウィンドウを非表示にする
        try:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        except AttributeError:
            pass  # Linux/Mac では不要

        self._proc = await asyncio.create_subprocess_exec(*self.command, **kwargs)

        # MCP ハンドシェイク: initialize → initialized 通知 → tools/list
        await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "streamlit-agent", "version": "1.0.0"},
            },
        )
        await self._notify("notifications/initialized", {})
        resp = await self._rpc("tools/list", {})
        self.tools = resp.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> Any:
        resp = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        return resp.get("result", {})

    async def _rpc(self, method: str, params: dict) -> dict:
        msg_id = str(uuid.uuid4())
        payload = (
            json.dumps(
                {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
            )
            + "\n"
        )
        async with self._lock:
            self._proc.stdin.write(payload.encode())
            await self._proc.stdin.drain()
            while True:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=30.0
                )
                if not line:
                    raise RuntimeError("EA MCP プロセスが終了しました")
                data = json.loads(line.decode().strip())
                if data.get("id") == msg_id:
                    return data

    async def _notify(self, method: str, params: dict):
        payload = (
            json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        )
        self._proc.stdin.write(payload.encode())
        await self._proc.stdin.drain()

    async def close(self):
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

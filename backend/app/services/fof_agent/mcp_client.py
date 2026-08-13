"""MCP (Model Context Protocol) client for FOF Agent.

Implements a stdio-based JSON-RPC MCP client that can connect to
external MCP servers and wrap their tools for use in the FOF Agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Definition of a tool from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPToolResult:
    """Result from executing an MCP tool."""

    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class MCPClient:
    """Client for connecting to MCP servers via stdio JSON-RPC."""

    def __init__(self, name: str, command: list[str], env: dict[str, str] | None = None) -> None:
        self.name = name
        self.command = command
        self.env = env or {}
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._tools: dict[str, MCPTool] = {}
        self._initialized = False

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._process is not None:
            return

        # Start the MCP server process
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**subprocess.os.environ.copy(), **self.env},
            text=True,
            bufsize=1,
        )

        # Initialize the connection
        await self._initialize()
        self._initialized = True
        logger.info(f"MCP client '{self.name}' connected with {len(self._tools)} tools")

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
        self._initialized = False

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request to the MCP server."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("MCP client not connected")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        # Write request
        request_str = json.dumps(request) + "\n"
        self._process.stdin.write(request_str)
        self._process.stdin.flush()

        # For simplicity, we'll do a blocking read here
        # In production, you'd want async I/O
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server disconnected")

        response = json.loads(response_line)

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        return response.get("result")

    async def _initialize(self) -> None:
        """Initialize the MCP connection and discover tools."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "fof-agent",
                "version": "1.0.0",
            },
        })

        # Send initialized notification
        await self._send_request("initialized")

        # List available tools
        tools_result = await self._send_request("tools/list")
        for tool in tools_result.get("tools", []):
            self._tools[tool["name"]] = MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
                server_name=self.name,
            )

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Call a tool on the MCP server."""
        import time as time_module

        if not self._initialized:
            await self.connect()

        start = time_module.perf_counter()
        try:
            result = await self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })

            return MCPToolResult(
                success=True,
                result=result,
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return MCPToolResult(
                success=False,
                error=str(e),
                duration_ms=(time_module.perf_counter() - start) * 1000,
            )

    def list_tools(self) -> list[MCPTool]:
        """List all available tools from this MCP server."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> MCPTool | None:
        """Get a specific tool by name."""
        return self._tools.get(name)


class MCPToolRegistry:
    """Registry for managing MCP server connections and tool wrappers."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPClient] = {}
        self._wrapped_tools: dict[str, Callable[..., Any]] = {}

    def add_server(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> MCPClient:
        """Add and connect to an MCP server."""
        client = MCPClient(name, command, env)
        self._servers[name] = client
        return client

    async def connect_all(self) -> None:
        """Connect to all registered MCP servers."""
        for client in self._servers.values():
            await client.connect()

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._servers.values():
            await client.disconnect()

    def wrap_tool(
        self,
        mcp_server: str,
        tool_name: str,
        wrapper_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Wrap an MCP tool for use in the FOF Agent."""

        async def wrapped_tool(**kwargs: Any) -> Any:
            client = self._servers.get(mcp_server)
            if client is None:
                raise ValueError(f"MCP server '{mcp_server}' not found")

            # Apply wrapper if provided
            if wrapper_fn:
                kwargs = wrapper_fn(kwargs)

            result = await client.call_tool(tool_name, kwargs)
            if not result.success:
                raise RuntimeError(f"MCP tool '{tool_name}' failed: {result.error}")

            return result.result

        self._wrapped_tools[tool_name] = wrapped_tool

    def get_wrapped_tool(self, name: str) -> Callable[..., Any] | None:
        """Get a wrapped tool by name."""
        return self._wrapped_tools.get(name)

    def list_all_tools(self) -> dict[str, MCPTool]:
        """List all tools from all connected MCP servers."""
        all_tools = {}
        for client in self._servers.values():
            for tool in client.list_tools():
                all_tools[f"{client.name}:{tool.name}"] = tool
        return all_tools


# MCP Server example (for reference)
# This would typically be a separate process/script
MCP_SERVER_TEMPLATE = '''#!/usr/bin/env python3
"""Example MCP server for FOF Agent.

This is a template for creating custom MCP servers that expose
external tools to the FOF Agent.
"""

import json
import sys

def handle_request(request):
    method = request.get("method")
    params = request.get("params", {})

    if method == "initialize":
        return {{
            "protocolVersion": "2024-11-05",
            "capabilities": {{}},
            "serverInfo": {{
                "name": "example-server",
                "version": "1.0.0"
            }}
        }}

    elif method == "tools/list":
        return {{
            "tools": [
                {{
                    "name": "example_tool",
                    "description": "An example tool",
                    "inputSchema": {{
                        "type": "object",
                        "properties": {{
                            "param1": {{"type": "string"}}
                        }},
                        "required": ["param1"]
                    }}
                }}
            ]
        }}

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {{}})

        if tool_name == "example_tool":
            # Tool implementation here
            return {{"result": f"Processed {{arguments.get('param1')}}"}}

    return {{"error": {{"code": -32601, "message": "Method not found"}}}}


if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip():
            request = json.loads(line)
            response = handle_request(request)
            if response:
                print(json.dumps(response))
'''

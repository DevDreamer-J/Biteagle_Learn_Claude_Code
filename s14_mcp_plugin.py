"""
s14_mcp_plugin.py - MCP 插件系统（动态发现 + 策略控制的外部工具）

    connect_mcp("docs")
              |
              v
    +------------------+     tools/list     +------------------+
    | Agent Harness    | <----------------- | MCP server       |
    |                  |                    | docs             |
    | built-in tools   |     tools/call     |                  |
    | + MCP tools      | -----------------> | search           |
    +--------+---------+                    | get_version      |
             |                              +------------------+
             v
    +-----------------------------------------------+
    | bash | read | write | edit | glob | connect  |
    | mcp__docs__search | mcp__docs__get_version   |
    +-----------------------------------------------+

MCP_HOST_POLICY 定义外部工具的 allow/confirm 权限。

迁移自 s14_mcp_plugin/code.py → OpenAI SDK。
"""

import glob
import json
import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------- 初始化 ----------
load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["OpenAI_BASE_URL"],
)
MODEL = os.environ["LLM_DEEPSEEK_MODEL"]
WORKDIR = Path.cwd()

ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax, and prefer dedicated file tools for file operations"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)
BASE_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use built-in and connected MCP tools to solve tasks. "
    "Call connect_mcp before using a server."
)


# ---------- Base Tools ----------

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, errors="replace", timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        output = output[:50000] if output else "(no output)"
        if result.returncode:
            return f"Error: command exited with status {result.returncode}\n{output}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except OSError as exc:
        return f"Error: {type(exc).__name__}: {exc}"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def run_write(path: str, content: str) -> str:
    try:
        target = (WORKDIR / path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        target = (WORKDIR / path).resolve()
        content = target.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count != 1:
            return f"Error: Expected 1 occurrence, found {count}"
        target.write_text(content.replace(old_text, new_text), encoding="utf-8")
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_glob(pattern: str) -> str:
    try:
        matches = sorted({
            match for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR.resolve())
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as exc:
        return f"Error: {exc}"


BASE_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read file contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace exact text once.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]},
    }},
    {"type": "function", "function": {
        "name": "glob", "description": "Find files by glob pattern; ** matches recursively.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    }},
]

BASE_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# ---------- MCP Discovery and Dispatch ----------

class MCPClient:
    """Small in-process stand-in for MCP tools/list and tools/call."""

    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict] = []
        self._handlers: dict[str, callable] = {}

    def register(self, tool_defs: list[dict], handlers: dict[str, callable]):
        names = [tool.get("name") for tool in tool_defs]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Every MCP tool needs a non-empty name")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate MCP tool name on server {self.name!r}")
        missing = [name for name in names if name not in handlers]
        if missing:
            raise ValueError(f"Missing MCP handlers: {', '.join(missing)}")
        self.tools = list(tool_defs)
        self._handlers = dict(handlers)

    def call_tool(self, tool_name: str, args: dict) -> str:
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return str(handler(**args))
        except Exception as exc:
            return f"MCP error: {type(exc).__name__}: {exc}"


mcp_clients: dict[str, MCPClient] = {}
mcp_tool_policies: dict[str, str] = {}
_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

# Authorization comes from host configuration, never server descriptions.
MCP_HOST_POLICY = {
    ("docs", "search"): "allow",
    ("docs", "get_version"): "allow",
    ("deploy", "status"): "allow",
    ("deploy", "trigger"): "confirm",
}


def normalize_mcp_name(name: str) -> str:
    """Replace characters outside the model tool-name alphabet."""
    normalized = _DISALLOWED_CHARS.sub("_", name)
    if not normalized:
        raise ValueError("MCP names cannot normalize to an empty string")
    return normalized


def _mock_server_docs() -> MCPClient:
    server = MCPClient("docs")
    server.register(
        tool_defs=[
            {
                "name": "search",
                "description": "Search the documentation.",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                "annotations": {"readOnlyHint": True},
            },
            {
                "name": "get_version",
                "description": "Get the documentation API version.",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": True},
            },
        ],
        handlers={
            "search": lambda query: f"[docs] Found 3 results for '{query}'",
            "get_version": lambda: "[docs] API v2.1.0",
        },
    )
    return server


def _mock_server_deploy() -> MCPClient:
    server = MCPClient("deploy")
    server.register(
        tool_defs=[
            {
                "name": "trigger",
                "description": "Trigger a deployment.",
                "inputSchema": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
                "annotations": {"destructiveHint": True},
            },
            {
                "name": "status",
                "description": "Check deployment status.",
                "inputSchema": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
                "annotations": {"readOnlyHint": True},
            },
        ],
        handlers={
            "trigger": lambda service: f"[deploy] Triggered: {service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        },
    )
    return server


MOCK_SERVERS = {
    "docs": _mock_server_docs,
    "deploy": _mock_server_deploy,
}


def connect_mcp(name: str) -> str:
    if name in mcp_clients:
        return f"MCP server '{name}' already connected"
    factory = MOCK_SERVERS.get(name)
    if not factory:
        return f"Unknown server '{name}'. Available: {', '.join(MOCK_SERVERS)}"
    server = factory()
    mcp_clients[name] = server
    names = ", ".join(tool["name"] for tool in server.tools)
    print(f"  [mcp] connected: {name} -> {names}")
    return (
        f"Connected to MCP server '{name}'. "
        f"Discovered {len(server.tools)} tools: {names}"
    )


def run_connect_mcp(name: str) -> str:
    return connect_mcp(name)


CONNECT_TOOL_OPENAI = {"type": "function", "function": {
    "name": "connect_mcp",
    "description": "Connect to an MCP server and discover its tools.",
    "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": ["docs", "deploy"]}}, "required": ["name"]},
}}

BUILTIN_TOOLS = [*BASE_TOOLS, CONNECT_TOOL_OPENAI]
BUILTIN_HANDLERS = {**BASE_HANDLERS, "connect_mcp": run_connect_mcp}


def assemble_tool_pool() -> tuple[list[dict], dict[str, callable]]:
    """Combine built-in tools with every connected server tool.

    迁移适配：MCP server 内部用 `inputSchema`（MCP 标准），
    OpenAI 用 `parameters`，这里做一层转换。
    """
    global mcp_tool_policies
    tools = list(BUILTIN_TOOLS)
    handlers = dict(BUILTIN_HANDLERS)
    policies: dict[str, str] = {}
    origins = {tool["function"]["name"]: f"built-in tool {tool['function']['name']!r}"
               for tool in tools}

    for server_name, server in mcp_clients.items():
        safe_server = normalize_mcp_name(server_name)
        for tool_def in server.tools:
            raw_name = tool_def["name"]
            safe_tool = normalize_mcp_name(raw_name)
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            if len(prefixed) > 64:
                raise ValueError(f"MCP tool name is longer than 64 characters: {prefixed}")
            origin = f"MCP tool {server_name!r}/{raw_name!r}"
            if prefixed in origins:
                raise ValueError(
                    "MCP tool name collision after normalization: "
                    f"{prefixed!r} maps both {origins[prefixed]} and {origin}"
                )
            # MCP inputSchema → OpenAI parameters
            schema = tool_def.get("inputSchema", {})
            if not isinstance(schema, dict) or schema.get("type", "object") != "object":
                raise ValueError(f"Invalid input schema for {origin}")
            origins[prefixed] = origin
            tools.append({"type": "function", "function": {
                "name": prefixed,
                "description": tool_def.get("description", ""),
                "parameters": schema,
            }})
            handlers[prefixed] = (
                lambda *, client=server, tool=raw_name, **kwargs:
                client.call_tool(tool, kwargs)
            )
            policies[prefixed] = MCP_HOST_POLICY.get((server_name, raw_name), "confirm")

    mcp_tool_policies = policies
    return tools, handlers


def assemble_system_prompt() -> str:
    if not mcp_clients:
        return BASE_SYSTEM
    return BASE_SYSTEM + "\n\nConnected MCP servers: " + ", ".join(mcp_clients)


# ---------- Hooks ----------

HOOKS: dict[str, list] = {
    "UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": [],
}
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


def permission_hook(tool_name: str, args: dict) -> str | None:
    """迁移后参数改为 (str, dict)；原参数是 Anthropic block。"""
    if tool_name == "bash":
        command = args.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                return f"Permission denied by deny list: {pattern}"
        if contains_destructive_command(command) or any(kw in command for kw in DESTRUCTIVE):
            print(f"\n[permission] {tool_name}({args})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"
    if tool_name in {"read_file", "write_file", "edit_file"}:
        raw_path = args.get("path", "")
        if not (WORKDIR / raw_path).resolve().is_relative_to(WORKDIR.resolve()):
            print(f"\n[permission] {tool_name}({args})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"
    if tool_name.startswith("mcp__"):
        policy = mcp_tool_policies.get(tool_name, "confirm")
        if policy != "allow":
            print(f"\n[permission] External tool {tool_name}({args})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"
    return None


def log_hook(tool_name: str, args: dict) -> None:
    preview = str(list(args.values())[:2])[:60]
    print(f"[hook] {tool_name}({preview})")


def large_output_hook(tool_name: str, args: dict, output: str) -> None:
    if len(str(output)) > 100000:
        print(f"[hook] Large output from {tool_name}: {len(str(output))} chars")


def context_hook(query: str):
    print(f"[hook] UserPromptSubmit: working in {WORKDIR}")
    return None


def summary_hook(messages: list):
    tool_count = sum(1 for message in messages if message.get("role") == "tool")
    print(f"[hook] Stop: session used {tool_count} tool calls")
    return None


register_hook("UserPromptSubmit", context_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def execute_tool(tool_name: str, args: dict, handlers: dict[str, callable]) -> str:
    blocked = trigger_hooks("PreToolUse", tool_name, args)
    if blocked:
        return str(blocked)
    handler = handlers.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}"
    try:
        output = str(handler(**args))
    except Exception as exc:
        output = f"Error: {type(exc).__name__}: {exc}"
    trigger_hooks("PostToolUse", tool_name, args, output)
    return output


# ---------- Agent loop with dynamic tool pool ----------

def agent_loop(messages: list):
    while True:
        try:
            tools, handlers = assemble_tool_pool()
            system_prompt = assemble_system_prompt()
            payload = [{"role": "system", "content": system_prompt}, *messages]
            response = client.chat.completions.create(
                model=MODEL, messages=payload, tools=tools, max_tokens=8000,
            )
        except Exception as exc:
            messages.append({
                "role": "assistant",
                "content": f"[Error] {type(exc).__name__}: {exc}",
            })
            trigger_hooks("Stop", messages)
            return

        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            trigger_hooks("Stop", messages)
            return

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"> {tc.function.name}")
            output = execute_tool(tc.function.name, args, handlers)
            print(output[:300])
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": output,
            })


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s14: MCP tools")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history: list[dict] = []

    while True:
        try:
            query = input("s14 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in {"q", "exit", ""}:
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]
        if last["role"] == "assistant" and last.get("content"):
            print(last["content"])
        print()

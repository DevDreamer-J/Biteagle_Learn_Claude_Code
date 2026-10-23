"""
s04_hooks.py - 生命周期钩子系统（s03 的硬编码权限 → Hook 机制）

四种钩子事件：

    UserPromptSubmit  —— 用户输入提交前
    PreToolUse        —— 工具执行前（返回非 None 即阻断）
    PostToolUse       —— 工具执行后（只观察，不阻断）
    Stop              —— 循环即将退出时（可返回字符串强制续跑）

核心抽象：agent loop 里只留一行 trigger_hooks，扩展逻辑全部注册成 callback。

迁移自 s04_hooks/code.py → OpenAI SDK。
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
    "Windows: bash tool uses cmd.exe syntax; prefer dedicated file tools "
    "(read_file/write_file/edit_file/glob) for file operations."
    if os.name == "nt"
    else "Unix-like: bash tool runs the system shell"
)

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use tools to solve tasks. Act, don't explain."
)


# ---------- 工具实现 ----------

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, errors="replace", timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    try:
        matches = sorted({
            match for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
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
        "name": "edit_file", "description": "Replace exact text in a file once.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]},
    }},
    {"type": "function", "function": {
        "name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    }},
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# ---------- s04 核心：Hook 系统 ----------

# 四种事件：UserPromptSubmit / PreToolUse / PostToolUse / Stop
HOOKS: dict[str, list] = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback):
    """注册回调到指定事件列表。"""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """依次触发指定事件的所有 callback。

    对于 PreToolUse / Stop，callback 返回非 None 值视为拦截：
      - PreToolUse：阻断本次工具调用，返回值作为 Permission denied 写入 tool 消息
      - Stop：强制循环不退出，返回值作为新的 user 消息追加
    """
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


# ---------- s03 的权限逻辑现在封装成 PreToolUse Hook ----------

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def _permission_hook(tool_name: str, args: dict) -> str | None:
    """PreToolUse：三級權限檢查。參數已解析（args 是 dict，不是 JSON 字符串）。"""
    if tool_name == "bash":
        command = args.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(kw in command for kw in DESTRUCTIVE):
            print(f"\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {tool_name}({args})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    if tool_name in ("read_file", "write_file", "edit_file"):
        path = args.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {tool_name}({args})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    return None


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


# ---------- 其他示例 Hooks ----------

def _log_hook(tool_name: str, args: dict) -> None:
    """PreToolUse：打印每个工具调用参数的前两个值作为预览。"""
    args_preview = str(list(args.values())[:2])[:60]
    print(f"\033[90m[HOOK] {tool_name}({args_preview})\033[0m")


def _large_output_hook(tool_name: str, args: dict, output: str) -> None:
    """PostToolUse：输出超过 100KB 警告（s03 的 run_bash 已截断到 50KB，所以这里一般不会触发）。"""
    if len(str(output)) > 100_000:
        print(f"\033[33m[HOOK] Large output from {tool_name}: {len(str(output))} chars\033[0m")


def _context_inject_hook(query: str) -> None:
    """UserPromptSubmit：提示用户当前工作目录。"""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")


def _summary_hook(messages: list) -> None:
    """Stop：打印本次会话用了多少次工具调用。"""
    # 从 messages 里数 role="tool" 的消息数
    tool_count = sum(1 for m in messages if m.get("role") == "tool")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")


# ---------- 注册所有 Hooks ----------
register_hook("UserPromptSubmit", _context_inject_hook)
register_hook("PreToolUse", _permission_hook)
register_hook("PreToolUse", _log_hook)
register_hook("PostToolUse", _large_output_hook)
register_hook("Stop", _summary_hook)


# ---------- Agent loop：hook 替换了硬编码的权限检查 ----------

def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
        )

        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            # Stop hook：可返回字符串强制续跑
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)

            # PreToolUse 链：permission + log 等。返回非 None 即阻断
            blocked = trigger_hooks("PreToolUse", tc.function.name, args)
            if blocked:
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"Permission denied: {blocked}",
                })
                continue

            handler = TOOL_HANDLERS.get(tc.function.name)
            output = handler(**args) if handler else f"Unknown: {tc.function.name}"

            # PostToolUse 链：只观察
            trigger_hooks("PostToolUse", tc.function.name, args, output)

            print(str(output)[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s04: Hooks - permission/log/summary as registered callbacks")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("s04 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]
        if last["role"] == "assistant" and last.get("content"):
            print(last["content"])
        print()

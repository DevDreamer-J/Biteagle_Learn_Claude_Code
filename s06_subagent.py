"""
s06_subagent.py - 子代理嵌套（task 工具启动独立 agent loop）

核心机制：
  Parent agent → 调用 task(prompt) → run_subagent()
  Subagent → 自己的消息列表 + SUB_TOOLS（只有 BASE_TOOLS，无 task）
  → 嵌套 agent loop 最多 30 轮 → 返回最终文本给父代理

关键设计：subagent 没有 task 工具，防止无限递归委派。

迁移自 s06_subagent/code.py → OpenAI SDK。
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

# 两套 system prompt：父代理和子代理各用一套
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use task for focused exploration or a self-contained subtask."
)
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Complete the given task, then return a concise final answer."
)


# ---------- 基础工具 ----------

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
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
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


# ---------- 工具定义 ----------

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
        "name": "edit_file", "description": "Replace exact text in a file once.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]},
    }},
    {"type": "function", "function": {
        "name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    }},
]

BASE_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

# 子代理工具集：只有 BASE_TOOLS，没有 task → 防止无限递归委派
SUB_TOOLS = list(BASE_TOOLS)
SUB_HANDLERS = dict(BASE_HANDLERS)


# ---------- Hook 系统 ----------

HOOKS: dict[str, list] = {
    "UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": [],
}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def _permission_hook(tool_name: str, args: dict) -> str | None:
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


def _log_hook(tool_name: str, args: dict) -> None:
    args_preview = str(list(args.values())[:2])[:60]
    print(f"\033[90m[HOOK] {tool_name}({args_preview})\033[0m")


def _large_output_hook(tool_name: str, args: dict, output: str) -> None:
    if len(str(output)) > 100_000:
        print(f"\033[33m[HOOK] Large output from {tool_name}: {len(str(output))} chars\033[0m")


def _context_inject_hook(query: str) -> None:
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")


def _summary_hook(messages: list) -> None:
    tool_count = sum(1 for m in messages if m.get("role") == "tool")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")


register_hook("UserPromptSubmit", _context_inject_hook)
register_hook("PreToolUse", _permission_hook)
register_hook("PreToolUse", _log_hook)
register_hook("PostToolUse", _large_output_hook)
register_hook("Stop", _summary_hook)


# ---------- execute_tool：公用于父 agent 和 subagent ----------

def execute_tool(tool_name: str, args: dict, handlers: dict) -> str:
    """跑 hook 链 + handler 执行，返回输出字符串。

    handlers 参数让父/子 agent 可以传不同的 handler 字典。
    """
    blocked = trigger_hooks("PreToolUse", tool_name, args)
    if blocked:
        return str(blocked)

    handler = handlers.get(tool_name)
    try:
        output = handler(**args) if handler else f"Unknown: {tool_name}"
    except Exception as e:
        output = f"Error: {e}"

    trigger_hooks("PostToolUse", tool_name, args, output)
    return str(output)


# ---------- s06 新增：Subagent ----------

def run_subagent(prompt: str) -> str:
    """启动嵌套 agent loop，最多 30 轮，返回最终文本。"""
    print("\n\033[35m[Subagent started]\033[0m")
    messages = [
        {"role": "system", "content": SUB_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    for _ in range(30):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=SUB_TOOLS,
        )

        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            trigger_hooks("Stop", messages)
            print("\033[35m[Subagent done]\033[0m")
            return msg.content or "(no summary)"

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            output = execute_tool(tc.function.name, args, SUB_HANDLERS)
            print(f"  \033[90m[sub] {tc.function.name}: {output[:100]}\033[0m")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

    print("\033[35m[Subagent stopped]\033[0m")
    return "Subagent stopped after 30 turns without a final answer."


# task 工具定义 + handler
TASK_TOOL = {"type": "function", "function": {
    "name": "task",
    "description": "Run a subagent with fresh conversation context and return its final text.",
    "parameters": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "minLength": 1}},
        "required": ["prompt"],
    },
}}

TOOLS = BASE_TOOLS + [TASK_TOOL]
TOOL_HANDLERS = {**BASE_HANDLERS, "task": run_subagent}


# ---------- 父 agent loop ----------

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
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            output = execute_tool(tc.function.name, args, TOOL_HANDLERS)
            print(str(output)[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s06: Subagent - task tool spawns nested agent loop")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("s06 >> ")
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

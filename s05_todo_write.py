"""
s05_todo_write.py - TodoManager + todo_write 工具（在 s04 基础上扩展）

新增：
  TodoManager   —— 状态容器，跟踪 pending / in_progress / completed 三态
  todo_write    —— LLM 调用此工具更新任务列表
  rounds_since_todo 计数器 —— 连续 3 轮未调用 todo_write 则注入 <reminder>

核心约束：最多 20 条，同时只能有 1 条 in_progress。

迁移自 s05_todo_write/code.py → OpenAI SDK。
"""

import ast     # ast.literal_eval 兜底解析 Python 列表字符串
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

# s05 变化：system prompt 引导 LLM 先规划再动手
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
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


# ---------- s05 新增：TodoManager ----------

class TodoManager:
    """结构化任务列表容器，LLM 通过 todo_write 工具更新。"""

    def __init__(self):
        self.items: list[dict] = []

    def update(self, todos: list | str) -> str:
        """接受 list 或 JSON 字符串 / Python 列表字符串，校验后替换 items。"""
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                # LLM 有时输出 Python 列表字面量，ast.literal_eval 安全兜底
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError) as e:
                    raise ValueError("todos must be a list or JSON array string") from e

        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        if len(todos) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] must be an object")

            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] has invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"content": content, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """渲染成 markdown 风格的复选框列表。"""
        if not self.items:
            return "No todos."

        lines = []
        for todo in self.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[todo["status"]]
            lines.append(f"{marker} {todo['content']}")

        done = sum(todo["status"] == "completed" for todo in self.items)
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


def run_todo_write(todos: list | str) -> str:
    """todo_write 工具 handler：更新 TodoManager 并打印面板。"""
    try:
        output = TODO.update(todos)
    except ValueError as e:
        return f"Error: {e}"
    print(f"\n\033[33m## Current Tasks\033[0m\n{output}")
    return output


# ---------- 工具定义（OpenAI function calling 格式） ----------

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

# s05 新增：todo_write 工具的 parameters 是嵌套 object（todos 是 object 数组）
TODO_TOOL = {"type": "function", "function": {
    "name": "todo_write",
    "description": "Create and manage a task list for your current coding session.",
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
}}

TOOLS = BASE_TOOLS + [TODO_TOOL]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
}


# ---------- Hook 系统（直接复用 s04） ----------

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


# ---------- Agent loop：新增 rounds_since_todo 计数器 ----------

def agent_loop(messages: list):
    rounds_since_todo = 0
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

        used_todo = False
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)

            blocked = trigger_hooks("PreToolUse", tc.function.name, args)
            if blocked:
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"Permission denied: {blocked}",
                })
                continue

            handler = TOOL_HANDLERS.get(tc.function.name)
            try:
                output = handler(**args) if handler else f"Unknown: {tc.function.name}"
            except Exception as e:
                output = f"Error: {e}"

            trigger_hooks("PostToolUse", tc.function.name, args, output)

            if tc.function.name == "todo_write":
                used_todo = True

            print(str(output)[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        # 计数：连续 N 轮没用 todo_write 就提醒
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            messages.append({
                "role": "user",
                "content": "<reminder>Update your todos.</reminder>",
            })
            rounds_since_todo = 0


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s05: TodoWrite - plan before execution")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("s05 >> ")
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

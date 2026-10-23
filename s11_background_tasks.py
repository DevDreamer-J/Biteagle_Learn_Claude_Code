"""
s11_background_tasks.py - 后台任务（独立线程 + 结果注入）

Main thread                              Background thread
+------------------------------+         +----------------------+
| bash(run_in_background=True) | ------> | run command          |
| return bg_id                 |         | queue result         |
| continue agent loop          | <------ +----------------------+
| next turn: collect           |
+------------------------------+

迁移自 s11_background_tasks/code.py → OpenAI SDK。
"""

import atexit
import glob
import json
import os
import re
import signal
import subprocess
import threading
import time
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
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use tools to solve tasks. "
    "Set run_in_background to true only for independent Bash commands."
)


# ---------- Shell 进程管理 ----------

_shell_processes: set[subprocess.Popen] = set()
_shell_process_lock = threading.RLock()


def _stop_process_group(process: subprocess.Popen):
    """Stop a shell process and its children (cross-platform).

    Windows: ``Popen.terminate()`` / ``Popen.kill()`` 没有 process group。
    POSIX: 用 SIGTERM 再 SIGKILL 杀整个进程组。
    """
    if os.name == "nt":
        for stop in (process.terminate, process.kill):
            if process.poll() is not None:
                return
            try:
                stop()
            except OSError:
                return
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                continue
        return

    # SIGKILL 只在 POSIX 上存在，没有就退回 SIGTERM
    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, OSError):
            return
        time.sleep(0.05)


def _stop_all_shell_processes():
    with _shell_process_lock:
        processes = list(_shell_processes)
    for process in processes:
        _stop_process_group(process)


def _handle_termination_signal(signum, _frame):
    _stop_all_shell_processes()
    raise SystemExit(128 + signum)


atexit.register(_stop_all_shell_processes)

# Windows 没有 SIGTERM，用 getattr 安全获取；CTRL_BREAK_EVENT 是 Windows 上能捕获的最接近的
_signal_to_handle = getattr(signal, "SIGTERM", None) or getattr(signal, "CTRL_BREAK_EVENT", None)
if _signal_to_handle is not None:
    try:
        signal.signal(_signal_to_handle, _handle_termination_signal)
    except (AttributeError, OSError):
        pass


def _run_bash_process(command: str) -> tuple[str, int | None]:
    process = None
    try:
        popen_kwargs = {} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen(
            command, shell=True, cwd=WORKDIR,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", **popen_kwargs,
        )
        with _shell_process_lock:
            _shell_processes.add(process)
        stdout, stderr = process.communicate(timeout=120)
        output = (stdout + stderr).strip()
        return (output[:50000] if output else "(no output)"), process.returncode
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)", None
    except OSError as error:
        return f"Error: {type(error).__name__}: {error}", None
    finally:
        if process is not None:
            _stop_process_group(process)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            with _shell_process_lock:
                _shell_processes.discard(process)


def _format_bash_result(output: str, exit_code: int | None) -> str:
    if exit_code in (0, None):
        return output
    return f"Error: command exited with status {exit_code}\n{output}"


def run_bash(command: str, run_in_background: bool = False) -> str:
    return _format_bash_result(*_run_bash_process(command))


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as error:
        return f"Error: {error}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as error:
        return f"Error: {error}"


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
    except Exception as error:
        return f"Error: {error}"


TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "run_in_background": {"type": "boolean"},
        }, "required": ["command"]},
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


# ---------- Hooks ----------

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


def permission_hook(tool_name: str, args: dict) -> str | None:
    if tool_name == "bash":
        command = args.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(kw in command for kw in DESTRUCTIVE):
            print("\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {tool_name}({args})")
            if input("   Allow? [y/N] ").strip().lower() not in ("y", "yes"):
                return "Permission denied by user"
    if tool_name in ("read_file", "write_file", "edit_file"):
        path = args.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {tool_name}({args})")
            if input("   Allow? [y/N] ").strip().lower() not in ("y", "yes"):
                return "Permission denied by user"
    return None


def log_hook(tool_name: str, args: dict) -> None:
    preview = str(list(args.values())[:2])[:60]
    print(f"\033[90m[HOOK] {tool_name}({preview})\033[0m")


def large_output_hook(tool_name: str, args: dict, output: str) -> None:
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] Large output from {tool_name}: {len(str(output))} chars\033[0m")


def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    tool_count = sum(1 for message in messages if message.get("role") == "tool")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def call_tool(tool_name: str, args: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    try:
        output = handler(**args) if handler else f"Unknown: {tool_name}"
    except Exception as error:
        output = f"Error: {error}"
    return str(output)


# ---------- Background 执行 ----------

class BackgroundManager:
    """后台线程管理器。每个后台 bash 命令独立起一个 daemon 线程。"""

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.results: dict[str, str] = {}
        self._ready: list[str] = []
        self._counter = 0
        self._lock = threading.Lock()

    def start(self, command: str, tool_call_id: str) -> str:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash command cannot be empty")

        with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"
            self.tasks[task_id] = {
                "tool_call_id": tool_call_id,
                "command": command,
                "status": "running",
            }

        thread = threading.Thread(
            target=self._run, args=(task_id, command), daemon=True,
        )
        try:
            thread.start()
        except Exception:
            with self._lock:
                self.tasks.pop(task_id, None)
            raise
        print(f"  [background] started {task_id}: {command[:60]}")
        return task_id

    def _run(self, task_id: str, command: str):
        try:
            output, exit_code = _run_bash_process(command)
            result = _format_bash_result(output, exit_code)
            status = "completed" if exit_code == 0 else "failed"
        except Exception as error:
            result = f"Error: {type(error).__name__}: {error}"
            status = "failed"

        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            task["status"] = status
            self.results[task_id] = result
            self._ready.append(task_id)

    def collect(self) -> list[str]:
        with self._lock:
            ready = []
            for task_id in self._ready:
                task = self.tasks.pop(task_id, None)
                result = self.results.pop(task_id, "")
                if task is not None:
                    ready.append((task_id, task, result))
            self._ready.clear()

        notifications = []
        for task_id, task, result in ready:
            notifications.append(
                f"<task_notification>\n"
                f"  <task_id>{task_id}</task_id>\n"
                f"  <status>{task['status']}</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{result[:500]}</summary>\n"
                f"</task_notification>"
            )
            print(f"  [background] collected {task_id}: {task['status']}")
        return notifications


BACKGROUND = BackgroundManager()
background_tasks = BACKGROUND.tasks
background_results = BACKGROUND.results


def should_run_background(tool_name: str, args: dict) -> bool:
    return tool_name == "bash" and args.get("run_in_background") is True


def start_background_task(command: str, tool_call_id: str) -> str:
    return BACKGROUND.start(command, tool_call_id)


def collect_background_results() -> list[str]:
    return BACKGROUND.collect()


def inject_background_results(messages: list) -> int:
    """把已完成的后台任务结果注入 messages。

    原始代码把后台通知作为 user 消息 content list 里的 text block（Anthropic 格式）。
    OpenAI 里 content 是字符串，这里直接追加一条 user 消息。
    """
    notifications = collect_background_results()
    if not notifications:
        return 0

    joined = "\n".join(notifications)
    messages.append({"role": "user", "content": joined})
    return len(notifications)


def execute_tool(tool_name: str, args: dict, tool_call_id: str = "") -> str:
    blocked = trigger_hooks("PreToolUse", tool_name, args)
    if blocked is not None:
        return str(blocked)

    if should_run_background(tool_name, args):
        try:
            command = args.get("command", "")
            task_id = start_background_task(command, tool_call_id)
            output = (
                f"[Background task {task_id} started] "
                "The result will be collected on a later turn."
            )
        except Exception as error:
            output = f"Error: {error}"
    else:
        output = call_tool(tool_name, args)

    trigger_hooks("PostToolUse", tool_name, args, output)
    return output


# ---------- Agent loop ----------

def agent_loop(messages: list):
    while True:
        inject_background_results(messages)
        payload = [{"role": "system", "content": SYSTEM}, *messages]
        response = client.chat.completions.create(
            model=MODEL, messages=payload, tools=TOOLS, max_tokens=8000,
        )
        msg = response.choices[0].message

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
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
            output = execute_tool(tc.function.name, args, tc.id)
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": output,
            })


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s11: Background Tasks - explicit background Bash execution")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history: list[dict] = []
    while True:
        try:
            query = input("s11 >> ")
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

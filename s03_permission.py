"""
s03_permission.py - 三级权限门禁（在 s02 基础上加 Permission Pipeline）

三级检查，每级都可能拒绝执行：

    Gate 1: Hard deny list     —— 永远禁止的命令（rm -rf /, sudo, shutdown...）
    Gate 2: Rule matching      —— 上下文规则（写 outside workspace、破坏性命令）
    Gate 3: User approval      —— 规则匹配后暂停，等用户 y/N

agent loop 里只多一行：

    if not check_permission(tc):
        continue

迁移自 s03_permission/code.py → OpenAI SDK。
"""

import glob     # glob 查找文件（复用 s02）
import json     # 解析 tool call 参数 JSON
import os       # 环境变量、工作目录
import re       # 正则匹配破坏性命令
import subprocess  # 执行 shell
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------- 初始化（同 s02） ----------
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

# s03 变化：system prompt 告诉 LLM 破坏性操作需要审批
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "All destructive operations require user approval."
)


# ---------- 工具实现（直接复用 s02，省略 safe_path 定义避免重复） ----------

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


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
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
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


# ---------- 工具定义 + 分发（同 s02） ----------

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


# ---------- s03 新增：三级权限 Pipeline ----------

# Gate 1: 硬编码黑名单 —— 出现即拒绝，不走后续检查
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]


def check_deny_list(command: str) -> str | None:
    """返回被命中的 deny reason（命中字符串），None 表示通过。"""
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: 规则匹配 —— 正则识别 rm / del 等破坏性命令（cmd.exe 下 del 是删除）
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


PERMISSION_RULES = [
    # 文件工具：路径逃逸 → 触发审批
    {"tools": ["read_file", "write_file", "edit_file"],
     "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "message": "Writing outside workspace"},
    # bash：破坏性命令 / 危险关键字 → 触发审批
    {"tools": ["bash"],
     "check": lambda args: contains_destructive_command(args.get("command", "")) or
                           any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"},
]


def check_rules(tool_name: str, args: dict) -> str | None:
    """遍历 PERMISSION_RULES，返回命中的 rule message，None 表示通过。"""
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# Gate 3: 用户审批 —— 规则匹配后交互询问 y/N
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# 三级管道组合：deny → rule → ask
def check_permission(tool_name: str, args: dict) -> str | None:
    """返回 None 表示允许执行，否则返回 deny 原因字符串。"""
    # Gate 1: 只有 bash 需要检查 deny list（破坏性 shell 命令）
    if tool_name == "bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\n\033[31m[blocked] {reason}\033[0m")
            return reason

    # Gate 2: 规则匹配
    reason = check_rules(tool_name, args)
    if reason:
        # Gate 3: 用户审批
        decision = ask_user(tool_name, args, reason)
        if decision == "deny":
            return reason

    return None


# ---------- Agent loop：s02 形状 + check_permission 拦截 ----------

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
            return

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"\033[36m> {tc.function.name}\033[0m")

            # s03 核心：执行前跑权限管道
            deny_reason = check_permission(tc.function.name, args)
            if deny_reason:
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"Permission denied: {deny_reason}",
                })
                continue

            handler = TOOL_HANDLERS.get(tc.function.name)
            output = handler(**args) if handler else f"Unknown: {tc.function.name}"
            print(str(output)[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s03: Permission - three-gate pipeline on top of s02")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("s03 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]
        if last["role"] == "assistant" and last.get("content"):
            print(last["content"])
        print()

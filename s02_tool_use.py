"""
s02_tool_use.py - 多工具分发（在 s01 agent loop 基础上扩展）

核心变化：新增四个文件工具 + TOOL_HANDLERS 分发字典

    bash       -> run_bash
    read_file  -> run_read
    write_file -> run_write
    edit_file  -> run_edit
    glob       -> run_glob
    safe_path  -> 防止文件路径逃逸工作目录

迁移自 s02_tool_use/code.py（Anthropic）→ OpenAI SDK。
"""

import glob     # glob.glob(pattern, root_dir=...) 查找匹配文件
import json     # 解析 LLM 返回的 tool call 参数 JSON 字符串
import os       # 环境变量、工作目录
import subprocess  # 执行 shell 命令
from pathlib import Path   # 跨平台路径处理

from dotenv import load_dotenv  # 加载 .env
from openai import OpenAI        # OpenAI 兼容接口 SDK


# ---------- 初始化 ----------
load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["OpenAI_BASE_URL"],
)
MODEL = os.environ["LLM_DEEPSEEK_MODEL"]

# 固定工作目录，safe_path 会强制所有文件操作在此之内
WORKDIR = Path.cwd()

# 告诉 LLM 运行环境：Windows 用 cmd.exe 语法，并且有专门的文件工具可替代 shell
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


# ---------- 基础工具：bash（来自 s01，WORKDIR 改为 Path 类型） ----------

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, errors="replace", timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ---------- 新增：safe_path + 四个文件工具 ----------

def safe_path(p: str) -> Path:
    """把用户给的路径拼到 WORKDIR 上并 resolve，检查是否逃逸工作目录。

    Windows 上 resolve() 会把大小写、符号链接、. .. 全部归一化后再比。
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    """读文件内容，可选 limit 限制前 N 行（调试大文件用）。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写文件，父目录不存在自动创建。"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的一段文本（只替换第一次出现）。

    比 write_file 安全：LLM 不需要把整个文件重新输出一遍，
    也避免了 LLM 输出截断导致的文件损坏。
    """
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
    """glob 模式查找文件，** 递归。结果去重并截断到 200 条。"""
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


# ---------- 工具定义：OpenAI function calling 格式 ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max lines to return (optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern; ** matches recursively.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]


# ---------- 工具分发字典：按名称查 handler，替代 s01 里硬编码的 run_bash ----------

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


def agent_loop(messages: list):
    """Agent 核心循环（与 s01 形状一致，仅工具调用分派改为 TOOL_HANDLERS）。"""
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
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
            print(f"\033[33m> {tc.function.name}\033[0m")

            # 按工具名字查 handler，**args 展开参数
            handler = TOOL_HANDLERS.get(tc.function.name)
            if handler:
                output = handler(**args)
            else:
                output = f"Error: Unknown tool {tc.function.name}"

            print(str(output)[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s02: Tool Use - five tools (bash + four file tools)")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("s02 >> ")
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

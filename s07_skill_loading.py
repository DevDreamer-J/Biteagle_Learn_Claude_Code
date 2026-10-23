"""
s07_skill_loading.py - 技能加载系统（SkillLoader + load_skill 工具）

启动时扫描 skills/<name>/SKILL.md，解析 YAML frontmatter，
把技能名 + 简短描述嵌入 system prompt（catalog）。
LLM 只在需要时调用 load_skill(name) 取完整 SKILL.md 内容。

依赖新增：pyyaml。
迁移自 s07_skill_loading/code.py → OpenAI SDK。
"""

import glob
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

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
SKILLS_DIR = WORKDIR / "skills"   # skills 目录与项目根平级

ENVIRONMENT_PROMPT = (
    "Windows: bash tool uses cmd.exe syntax; prefer dedicated file tools "
    "(read_file/write_file/edit_file/glob) for file operations."
    if os.name == "nt"
    else "Unix-like: bash tool runs the system shell"
)


# ---------- SkillLoader ----------

class SkillLoader:
    """启动时扫描 skills/ 目录，构建技能 catalog。

    目录结构：
        skills/
          code-review/SKILL.md     ← 文件名固定
          pdf/SKILL.md

    SKILL.md 格式（可选 frontmatter）：
        ---
        name: code-review
        description: 代码审查技能
        ---

        # code-review
        具体内容...
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict[str, str]] = {}
        self.scan()

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict, str]:
        """解析 YAML frontmatter（--- 之间的部分），返回 (metadata, body)。

        无 frontmatter 或 YAML 解析失败时返回 ({}, 原文)。
        """
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\r\n") != "---":
            return {}, text

        closing_index = next(
            (index for index, line in enumerate(lines[1:], start=1)
             if line.rstrip("\r\n") == "---"),
            None,
        )
        if closing_index is None:
            return {}, text

        frontmatter = "".join(lines[1:closing_index])
        body = "".join(lines[closing_index + 1:]).strip()
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata, body

    def scan(self):
        """扫描 skills_dir 下所有 SKILL.md，填充 self.skills。"""
        self.skills.clear()
        if not self.skills_dir.exists():
            return

        skills_root = self.skills_dir.resolve()
        for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
            if (not manifest.is_file()
                    or not manifest.resolve().is_relative_to(skills_root)):
                continue

            content = manifest.read_text(encoding="utf-8")
            metadata, body = self.parse_frontmatter(content)

            # 名称优先用 frontmatter 的 name，否则用目录名
            raw_name = metadata.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            name = name or manifest.parent.name

            # 描述优先用 frontmatter 的 description，否则用正文第一行（去掉 # 标记）
            raw_description = metadata.get("description")
            description = raw_description.strip() if isinstance(raw_description, str) else ""
            description = description or body.split("\n", 1)[0]
            description = " ".join(str(description).lstrip("# ").split())

            self.skills[name] = {
                "name": name,
                "description": description,
                "content": content,
            }

    def catalog(self) -> str:
        """渲染 catalog，嵌入 system prompt。"""
        if not self.skills:
            return "(no skills found)"
        return "\n".join(
            f"- {skill['name']}: {skill['description']}"
            for skill in self.skills.values()
        )

    def load(self, name: str) -> str:
        """按名称取完整 SKILL.md 内容。"""
        skill = self.skills.get(name)
        if skill:
            return skill["content"]
        available = ", ".join(self.skills) or "none"
        return f"Error: Unknown skill '{name}'. Available: {available}"


SKILL_LOADER = SkillLoader(SKILLS_DIR)


# ---------- build_system_prompt：把技能 catalog 嵌入 system prompt ----------

def build_system_prompt() -> str:
    return (
        f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
        "Use tools to solve tasks. Act, don't explain.\n\n"
        f"Skills available:\n{SKILL_LOADER.catalog()}\n\n"
        "Use load_skill to read the full instructions when a skill applies."
    )


SYSTEM = build_system_prompt()


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

# s07 新增：load_skill 工具
LOAD_SKILL_TOOL = {"type": "function", "function": {
    "name": "load_skill",
    "description": "Load the full SKILL.md content by skill name.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}}

TOOLS = BASE_TOOLS + [LOAD_SKILL_TOOL]
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
    "load_skill": SKILL_LOADER.load,  # 直接引用 SkillLoader.load 方法
}


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


# ---------- execute_tool + agent loop ----------

def execute_tool(tool_name: str, args: dict) -> str:
    blocked = trigger_hooks("PreToolUse", tool_name, args)
    if blocked:
        return str(blocked)
    handler = TOOL_HANDLERS.get(tool_name)
    try:
        output = handler(**args) if handler else f"Unknown: {tool_name}"
    except Exception as e:
        output = f"Error: {e}"
    trigger_hooks("PostToolUse", tool_name, args, output)
    return str(output)


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
            output = execute_tool(tc.function.name, args)
            print(str(output)[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s07: Skill Loading - catalog first, full content on demand")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("s07 >> ")
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

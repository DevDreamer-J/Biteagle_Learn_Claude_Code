"""
s01_agent_loop.py - 用 OpenAI SDK 实现的最简 Agent Loop

本文件演示 AI Coding Agent 的核心模式：LLM 决定是否调用工具，
工具执行结果反馈给 LLM，循环往复直到 LLM 不再请求工具。

核心模式（伪代码）：
    while True:
        response = LLM(messages, tools)
        if response has no tool_calls:
            break                       # LLM 认为任务完成，输出最终回答
        execute tools                    # 按顺序执行每个工具
        append results                   # 把执行结果作为 tool 消息追加

消息格式（OpenAI Chat Completions）：
    system  —— 系统提示词（首条消息）
    user    —— 用户输入
    assistant —— LLM 回复（含可选的 tool_calls）
    tool    —— 工具执行结果，携带 tool_call_id 与 assistant 的 tool_calls 对应

依赖：
    pip install python-dotenv openai
    .env 需包含 API_KEY / OpenAI_BASE_URL / LLM_DEEPSEEK_MODEL

用法：
    python s01_agent_loop.py
    进入交互提示符后输入任务，输入 q 或回车退出。
"""

import json     # 解析 LLM 返回的 tool call 参数（JSON 字符串 → dict）
import os       # 读取环境变量、获取当前工作目录
import subprocess  # 执行 shell 命令并捕获输出

from dotenv import load_dotenv  # 加载 .env 文件到 os.environ
from openai import OpenAI        # OpenAI 兼容接口 SDK


# ---------- 初始化 ----------
# 强制从 .env 覆盖已有环境变量，避免系统级同名变量干扰
load_dotenv(override=True)

# 初始化 OpenAI 客户端。此项目使用阿里云百炼的 OpenAI 兼容接口，
# base_url 末尾带 /compatible-mode/v1，API Key 由百炼控制台提供。
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["OpenAI_BASE_URL"],
)

# 模型名：.env 中 LLM_DEEPSEEK_MODEL 键，默认值 deepseek-v4.1-flash-1017
MODEL = os.environ["LLM_DEEPSEEK_MODEL"]

# 系统提示词：注入当前工作目录让 LLM 知道自己在哪个位置执行；
# "Act, don't explain" 引导 LLM 直接调用工具而非长篇规划。
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to solve tasks. Act, don't explain."
)


# ---------- 工具定义 ----------
# OpenAI function calling 格式：每个工具是一个 dict，
# "type": "function" + "function" 对象（含 name / description / parameters JSON Schema）。
# 这里只暴露一个 bash 工具，让 LLM 通过 shell 命令完成一切。
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
    }
]


def run_bash(command: str) -> str:
    """执行 shell 命令并返回输出，带安全拦截与超时保护。

    Args:
        command: 待执行的 shell 命令字符串。

    Returns:
        命令的 stdout + stderr 合并内容（最多 50000 字符）；
        或以 "Error: ..." 开头的错误描述。
    """
    # 危险命令黑名单：拦截不可逆或破坏性操作，返回错误而非执行
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # shell=True 让系统 shell 解析命令（支持管道、重定向等）；
        # cwd=os.getcwd() 保证命令在当前目录执行，与 LLM 感知的工作目录一致；
        # timeout=120s 防止卡死；capture_output=True 同时捕获 stdout 和 stderr。
        r = subprocess.run(
            command, shell=True, cwd=os.getcwd(),
            capture_output=True, text=True, errors="replace", timeout=120,
        )
        # 合并 stdout 与 stderr，去掉首尾空白；
        # 超过 50000 字符截断，避免把超长输出塞回消息上下文导致 token 爆炸。
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def agent_loop(messages: list):
    """Agent 核心循环：调用 LLM → 检测工具请求 → 执行 → 回传结果，直到 LLM 停止。

    循环终止条件：LLM 返回的 message 没有 tool_calls 字段。
    此时 messages 列表末尾是 LLM 的最终文本回答，由调用方负责展示。

    Args:
        messages: 完整的消息历史列表，首轮调用前应已包含 system 和 user 消息。
                  本函数会原地追加 assistant / tool 消息，调用方可直接复用。
    """
    while True:
        # 调用 LLM。messages 是累积的完整上下文，LLM 据此决定下一步动作；
        # tools 参数告诉 LLM 当前可用的工具清单。
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        # 取第一条候选的 message（单候选请求只需关注 choices[0]）
        msg = response.choices[0].message

        # 把 assistant 消息追加到历史。tool_calls 存在时一并记录，
        # 但要把 Pydantic 对象序列化为 dict，保证下一轮发送给 SDK 时是纯 JSON 结构。
        assistant_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_msg)

        # 终止条件：LLM 没有请求任何工具 → 认为任务已完成
        if not msg.tool_calls:
            return

        # 执行每个 tool call 并追加 tool 消息。
        # 多工具场景（parallel tool calls）下，按 LLM 返回顺序串行执行即可。
        for tc in msg.tool_calls:
            # LLM 返回的 tool call 参数是 JSON 字符串，需要 json.loads 还原为 dict
            args = json.loads(tc.function.arguments)

            # 彩色打印命令本身，方便人类观察 Agent 在做什么
            print(f"\033[33m$ {args['command']}\033[0m")

            # 执行命令并拿到输出
            output = run_bash(args["command"])

            # 输出前 200 字符作为预览（run_bash 内部已截断到 50000，
            # 这一层是控制台展示的二次截断，防止刷屏）
            print(output[:200])

            # 把执行结果作为 role="tool" 的消息追加到历史，
            # tool_call_id 必须和对应 assistant 消息里的 tool_calls[].id 精确匹配，
            # LLM 才能把结果和自己之前的请求对应上。
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })


# ---------- 入口 ----------
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，按回车键发送。输入 q 退出。\n")

    # messages 历史以 system 消息开头，这是 OpenAI Chat Completions 的约定。
    history = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            query = input("s01 >> ")
        except (EOFError, KeyboardInterrupt):
            break                       # Ctrl-D / Ctrl-C 优雅退出
        if query.strip().lower() in ("q", "exit", ""):
            break                       # 空输入或退出命令 → 退出交互

        # 追加本轮用户输入，让 agent_loop 接管
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # agent_loop 返回后，history 末尾就是 LLM 的最终回答，打印出来
        last = history[-1]
        if last["role"] == "assistant" and last.get("content"):
            print(last["content"])
        print()

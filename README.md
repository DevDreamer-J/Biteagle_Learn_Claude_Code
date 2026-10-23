# learn-claude-code（OpenAI SDK 版）

用 17 个渐进式章节，从零构建一个能干活的 Coding Agent。本项目是 `shareAI-lab/learn-claude-code` 17 章主线的 **OpenAI SDK 移植版**——原版使用 Anthropic SDK（`claude` 模型），本版全部改用 OpenAI SDK 调用 OpenAI 兼容接口（阿里云百炼），并对 Windows 平台做了适配。

## 核心理念

**Agent = 模型 + Harness。** 模型负责决策，Harness 负责给模型双手、双眼、工作空间和边界。

s01 用最少的代码写出一个最小 Agent 循环，之后 16 章每一章只做一件事：在这个循环外面再加一层 Harness 机制。**循环本身几乎从不改动，变的只是循环周围挂载的东西。**

```
用户消息 -> messages[] -> LLM -> response
                                 |
                          有 tool_calls?
                         /              \
                       有                没有
                       |                  |
                 执行工具             返回文本
                 追加 role="tool" 消息
                 回到循环
```

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

或使用 uv 免安装直接运行：

```powershell
uv run --with python-dotenv --with openai --with pyyaml python s01_agent_loop.py
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入三个必需项：

| 键 | 说明 |
|---|---|
| `API_KEY` | API Key |
| `OpenAI_BASE_URL` | OpenAI 兼容接口地址（末尾 `/compatible-mode/v1`） |
| `LLM_DEEPSEEK_MODEL` | 主模型 ID |

### 3. 运行

s01–s15 均为交互式 REPL（输入 `q` 退出）：

```powershell
python s01_agent_loop.py
```

s16 / s17 支持命令行参数：

```powershell
python s16_workflow_runtime.py demo     # 或 resume / 无参数进入交互
python s17_goal_loop.py                 # asyncio 入口
```

## 章节地图

| 章节 | 文件 | 主题 | 新增机制 |
|---|---|---|---|
| s01 | [s01_agent_loop.py](s01_agent_loop.py) | 最小 Agent 循环 | 单 bash 工具 + while 循环 |
| s02 | [s02_tool_use.py](s02_tool_use.py) | 多工具 | 5 个工具 + `TOOL_HANDLERS` 分发 + `safe_path` 路径逃逸防护 |
| s03 | [s03_permission.py](s03_permission.py) | 权限门禁 | 三级管线：deny list → 规则匹配 → 用户确认 |
| s04 | [s04_hooks.py](s04_hooks.py) | 生命周期钩子 | 四事件 Hook 链（UserPromptSubmit / PreToolUse / PostToolUse / Stop） |
| s05 | [s05_todo_write.py](s05_todo_write.py) | 任务清单 | `TodoManager`（≤20 条、单 in_progress）+ 遗忘提醒注入 |
| s06 | [s06_subagent.py](s06_subagent.py) | 子代理 | task 工具派生嵌套 Agent 循环（子代理无 task 工具防递归） |
| s07 | [s07_skill_loading.py](s07_skill_loading.py) | 技能加载 | 扫描 `skills/*/SKILL.md`，catalog 进 system prompt，按需加载全文 |
| s08 | [s08_context_compact.py](s08_context_compact.py) | 上下文压缩 | 四层 pipeline：截断 → snip → micro → LLM 摘要 |
| s09 | [s09_memory.py](s09_memory.py) | 记忆系统 | `.memory/` 持久化 + MEMORY.md 索引 + LLM 选择读写 |
| s10 | [s10_task_system.py](s10_task_system.py) | 任务系统 | `.tasks/` JSON 持久化 + DAG 依赖 + 状态机 |
| s11 | [s11_background_tasks.py](s11_background_tasks.py) | 后台任务 | bash `run_in_background` + threading 后台执行 |
| s12 | [s12_cron_scheduler.py](s12_cron_scheduler.py) | 定时调度 | cron 表达式 + 定时触发提示词 |
| s13 | [s13_agent_teams.py](s13_agent_teams.py) | 多 Agent 团队 | MessageBus + Mailbox 团队通信 |
| s14 | [s14_mcp_plugin.py](s14_mcp_plugin.py) | MCP 插件 | connect_mcp + MCP 工具发现接入 |
| s15 | [s15_integrated_harness.py](s15_integrated_harness.py) | 综合集成 | 26 个内置工具，前 14 章机制合一 |
| s16 | [s16_workflow_runtime.py](s16_workflow_runtime.py) | 工作流编排 | agent / parallel / pipeline 编排 + journal 断点恢复 |
| s17 | [s17_goal_loop.py](s17_goal_loop.py) | 目标循环 | GoalController + 独立评估器模型驱动目标达成 |

建议按编号顺序学习：s01–s04 打基础（循环、工具、权限、钩子），s05–s09 加数据机制（清单、子代理、技能、压缩、记忆），s10–s14 加运行时能力（任务、后台、定时、团队、插件），s15–s17 综合与编排。

## 与原版（Anthropic SDK）的差异

- **SDK 调用**：`client.messages.create()` → `client.chat.completions.create()`；system prompt 从顶层参数移入 `messages` 首条 `role="system"` 消息
- **工具定义**：改为 OpenAI function 格式 `{"type": "function", "function": {"name", "description", "parameters"}}`
- **工具调用检测**：Anthropic content block（`tool_use`）→ OpenAI 扁平结构（assistant 消息的 `tool_calls` 字段）
- **参数解析**：`block.input` → `json.loads(tc.function.arguments)`
- **工具结果**：content block（`tool_result`）→ 独立 `role="tool"` 消息 + `tool_call_id` 匹配
- **Windows 适配**：`fcntl` 文件锁 → `threading.Lock` 退化；`signal.SIGTERM` 等 Unix 特性做平台保护

## 配套文档

- [s01-s17-学习指南.md](s01-s17-学习指南.md)：逐章深入讲解每个机制的设计动机、边界与代价（基于原版仓库编写，代码片段为 Anthropic 风格，概念与本版一一对应）

## 注意事项

- 所有示例直接读取项目根目录的 `.env`，键名不匹配会在启动时报错
- s07 的技能加载需要 `skills/*/SKILL.md` 目录（未提供示例时 catalog 为空，不影响运行）
- s09 / s10 会在运行目录下创建 `.memory/`、`.tasks/` 持久化目录
- 运行日志与临时产物写入 `.runtime/`

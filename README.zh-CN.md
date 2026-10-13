# Voice Task Agent

支持语音输入的任务型 Agent 框架，集成 Whisper 转写、任务规划、工具执行和多轮会话管理。

[English](README.md) · [架构说明](docs/ARCHITECTURE.md)

## 核心功能

- 统一结构化模型接口，对接云端 API 和兼容 Chat Completions 的本地服务。
- 每轮任务规划与 ReAct 执行，支持时间预算、步数限制和输出修复。
- 工具参数校验、名称白名单和重复调用检测。
- SQLite 会话持久化、版本检查、原子提交和请求重放。
- Whisper 音频文件转写与麦克风输入。
- 面试助手、本地笔记问答和自定义任务示例。

## 快速运行

需要 Python 3.11 或更新版本：

```bash
python -m pip install -e ".[dev]"
voice-agent --demo --app interviewer --context examples/interviewer.json --text "开始面试"
voice-agent --demo --app knowledge --context examples/knowledge.json --text "What is Whisper?"
python examples/custom_task.py
```

`--demo` 使用确定性的离线响应。省略 `--text` 可进入交互模式。

## 模型配置

在 shell 中设置 `AGENT_BASE_URL`、`AGENT_MODEL`、`AGENT_API_KEY`，去掉 `--demo`
即可调用模型。配置见 [`.env.example`](.env.example) 和 [英文说明](README.md#model-configuration)。
`AGENT_RESPONSE_FORMAT` 支持 `json_object`（默认）、`json_schema` 和 `text`。

## 语音输入

```bash
python -m pip install -e ".[voice]"
voice-agent --app interviewer --context examples/interviewer.json --record 5 --language zh
voice-agent --app knowledge --context examples/knowledge.json --audio question.wav
```

Whisper 按完整音频片段转写，默认使用 CPU/int8。`--asr-model` 可指定模型名称或本地目录。
麦克风录音需要 PortAudio 和系统权限。GPU 配置见 [英文说明](README.md#voice-input)。

## 会话与扩展

使用相同的 `--app` 和 `--db`，通过 `--session SESSION_ID` 继续会话。
使用相同请求 ID 和文本可重放已完成的结果。任务配置或工具结构变化后需创建新会话。

进程崩溃后，先停止旧进程，再释放未完成请求：

```bash
voice-agent --db sessions.sqlite3 --session SESSION_ID --recover
```

通过 `TaskSpec` 定义目标、指令和预算，通过 `ToolRegistry` 注册带类型参数的异步只读工具，
再提供会话上下文即可接入新任务。参考 [自定义任务示例](examples/custom_task.py)。

## 开发

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

## 代码来源

Agent 组件基于 [Fujizzz/AiInterviewer](https://github.com/Fujizzz/AiInterviewer)。

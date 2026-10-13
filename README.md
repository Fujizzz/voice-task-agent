# Voice Task Agent

A Python framework for voice-enabled task agents, combining Whisper transcription,
task planning, tool execution, and persistent multi-turn sessions.

[中文说明](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md)

## Features

- Structured model interface for hosted APIs and local Chat Completions services.
- Per-turn planning and ReAct execution with time and step budgets.
- Typed read-only tools, argument validation, and repeated-call detection.
- SQLite sessions with version checks, atomic commits, and request replay.
- Whisper audio-file and microphone input through `faster-whisper`.
- Interview, local-notes, and custom task examples.

## Quick start

Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
voice-agent --demo --app knowledge --context examples/knowledge.json --text "What is Whisper?"
voice-agent --demo --app interviewer --context examples/interviewer.json --text "Start the interview"
python examples/custom_task.py
```

`--demo` uses deterministic offline responses. Omit `--text` for interactive input.
Results include `session_id`, `state_version`, `plan`, and execution metadata.

## Model configuration

Export the variables in [`.env.example`](.env.example) in your shell. The CLI reads
environment variables directly.

```bash
export AGENT_BASE_URL="https://api.openai.com/v1"
export AGENT_MODEL="your-model-name"
export AGENT_API_KEY="your-key"
voice-agent --app interviewer --context examples/interviewer.json --text "Start the interview"
```

For a local compatible server, set `AGENT_BASE_URL` to its API root, such as
`http://127.0.0.1:8000/v1`, and `AGENT_MODEL` to the served model name.

`AGENT_RESPONSE_FORMAT` supports `json_object` (default), `json_schema`, and `text`.
Responses are validated with Pydantic.

## Voice input

```bash
python -m pip install -e ".[voice]"
voice-agent --app knowledge --context examples/knowledge.json --audio question.wav
voice-agent --app interviewer --context examples/interviewer.json --record 5 --language zh
```

Whisper transcribes complete audio segments using CPU/int8 by default.
`--asr-model` accepts a model name or local directory. GPU inference uses
`--asr-device cuda --asr-compute-type float16` with a compatible CTranslate2/CUDA environment.
Microphone recording requires PortAudio and microphone permission; temporary recordings
are removed on exit. Add `--demo` to pair voice input with offline responses.

## Sessions

Resume a session with the same `--app` and `--db`, reusing its saved context:

```bash
voice-agent --app knowledge --db sessions.sqlite3 --session SESSION_ID --text "Continue"
```

Reuse `--request-id` with identical text to replay a completed turn. Requests are
serialized within each session; different sessions can run concurrently. Task configuration
and tool-schema changes require a new session.

After a process crash, stop the old worker and release its pending request:

```bash
voice-agent --db sessions.sqlite3 --session SESSION_ID --recover
```

## Custom tasks

Define a `TaskSpec`, register typed async tools with `ToolRegistry`, and provide session context.
Implement `ModelPort` to connect a model. See [the custom task example](examples/custom_task.py)
and [the architecture guide](docs/ARCHITECTURE.md).

## Development

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

GitHub Actions runs tests on Windows and Linux with Python 3.11 and 3.12.
Tests use injected model and ASR responses.

## Attribution

Agent components are based on [Fujizzz/AiInterviewer](https://github.com/Fujizzz/AiInterviewer).

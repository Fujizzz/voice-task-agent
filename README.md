# Voice Task Agent

A small Python framework for task-oriented dialogue with optional local Whisper input.
The same runtime drives an interview assistant and a local-notes assistant. Planning,
bounded tool execution, model transport, and durable session state are separate components.

This repository refactors the Agent design from [Fujizzz/AiInterviewer](https://github.com/Fujizzz/AiInterviewer).
See [source provenance](docs/PROVENANCE.md) for the exact upstream revision and extraction map.

[中文说明](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Validation](docs/VALIDATION.md)

## What it implements

- A provider-neutral structured model interface and an OpenAI-compatible Chat Completions adapter.
- Per-turn planning followed by a bounded ReAct-style action/observation loop.
- Typed, allowlisted, read-only tools; argument validation; repeated-call detection.
- Shared turn deadline, per-call timeouts, bounded output repair, and explicit fallback results.
- SQLite sessions, version checks, atomic turn commits, and persistent request replay.
- Local Whisper transcription through optional `faster-whisper`, audio-file input, and explicit microphone recording.
- Two task applications plus a third example demonstrating the public extension interface.
- Offline tests without paid API requests, downloaded model weights, or microphone access.

## Quick start (no API key)

Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
voice-agent --demo --app knowledge --context examples/knowledge.json --text "What is Whisper?"
voice-agent --demo --app interviewer --context examples/interviewer.json --text "Start the interview"
python examples/custom_task.py
```

`--demo` uses a deterministic demonstration adapter, not an LLM. It verifies application wiring,
tool observations, and persistence; it does not demonstrate model intelligence or answer quality.
The JSON result includes `session_id`, `state_version`, `plan`, and execution metadata.

Omit `--text` for interactive text input. Resume an existing task with `--session SESSION_ID`
and the same `--app` / `--db`; do not supply a replacement context. To retry one turn without
another model call, reuse its `--request-id` and identical text.

## Hosted APIs and local inference

Configure an endpoint that implements `/chat/completions`. For example, in PowerShell:

```powershell
$env:AGENT_BASE_URL = 'https://api.openai.com/v1'
$env:AGENT_MODEL = 'your-model-name'
$env:AGENT_API_KEY = 'your-key'
voice-agent --app interviewer --context examples/interviewer.json --text 'Start the interview'
```

For a compatible local server (for example one exposed by vLLM or Ollama):

```powershell
$env:AGENT_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:AGENT_MODEL = 'your-served-model-name'
$env:AGENT_API_KEY = 'EMPTY'
voice-agent --app knowledge --context examples/knowledge.json --text 'Explain task planning'
```

Set `AGENT_RESPONSE_FORMAT` to `json_object` (default), `json_schema`, or `text` according to
your server's support. Every response is validated locally with Pydantic. The adapter does not
silently change providers or retry billable requests. No specific model deployment or hosted
provider is certified by the offline tests. `.env.example` is a template; export variables in
your shell or use your application's environment loader.

## Voice input

```bash
python -m pip install -e ".[voice]"
voice-agent --demo --app knowledge --context examples/knowledge.json --audio question.wav
voice-agent --demo --app interviewer --context examples/interviewer.json --record 5 --language zh
```

Drop `--demo` to combine Whisper with a configured real LLM. `--demo` replaces only the LLM;
it still uses real ASR when `--audio` or `--record` is selected. Recording starts only with
the explicit `--record` option. Temporary microphone recordings are removed on exit.

Whisper runs on CPU/int8 by default. `--asr-model` accepts a model name (may download weights)
or local model directory. Set `--asr-device cuda --asr-compute-type float16` only with a
compatible CTranslate2/CUDA environment. Microphone capture needs OS permission and a working
PortAudio installation. Recognition is utterance-based: no streaming ASR, wake word, TTS,
speaker diarization, or interruption/barge-in is claimed in this release.

ASR has its own timeout before the Agent turn starts. Cancelling the async wrapper does not
forcibly stop inference already running in a worker thread. Audio retries may repeat ASR;
use the finalized text and a stable request ID when exact replay is required.

## Define another task

```python
from voice_task_agent import TaskSpec, ToolRegistry, AgentRuntime, SQLiteStore
from voice_task_agent.providers import DemoModel

task = TaskSpec(
    name="clarification-assistant",
    instructions="Ask concise questions to clarify the user's requirements.",
    goal="Produce an agreed set of requirements.",
)
agent = AgentRuntime(DemoModel(), SQLiteStore(), task, ToolRegistry())
session_id = agent.create_session({"domain": "document processing"})
# result = await agent.run(session_id, "Help me scope a document parser")
```

Use a real `ModelPort` for meaningful generation. [The custom task example](examples/custom_task.py)
shows an async tool with typed arguments. Application context and tool results are data, not
system instructions. Prompt boundaries reduce accidental instruction mixing but are not a
complete defense against prompt injection; the runtime's tool allowlist is enforced in code.

## Session behavior

SQLite is a local library storage backend, not a hosted multi-tenant service. Applications
exposing it over a network must add authentication and authorization. Transcript text is
stored in the local database; model traces contain status/timing/tool names, not full prompts
or credentials. Core database/audio outputs are ignored by Git.

One request at a time may run per session, including across runtime instances. Different
sessions can execute concurrently. Completed requests are replayable after restart. A process
crash leaves its request reserved rather than silently launching duplicate work. Stop the old
worker before explicit recovery:

```bash
voice-agent --db sessions.sqlite3 --session SESSION_ID --recover
```

The v0.1 registry accepts read-only tools only. Exactly-once external writes are not provided.
Durable replay covers committed Agent turns; a cancelled or crashed uncommitted turn may be
recomputed. Changes to task configuration/tool schemas require a new session. Context is
immutable for a session; history retained for prompting is bounded.

## Test and package

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

GitHub Actions runs core tests on Windows and Linux with Python 3.11 and 3.12. Voice adapters
are tested with injected ASR results; real recognition accuracy, microphone operation, GPU
behavior, and paid/local LLM performance need environment-specific acceptance testing.

## Scope and source

The interviewer is a compact framework example, not a migration of AiInterviewer's full
scoring, UI, database, or recommendation product. The notes assistant uses simple lexical
retrieval, not a production vector RAG service. This repository records a current refactor;
it does not reconstruct historical environments or backdate development.

The reviewed upstream snapshot has no root license file. No new blanket license is assigned
to inherited material here; review ownership and licensing before third-party redistribution.

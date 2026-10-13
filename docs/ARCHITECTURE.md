# Architecture

The async Python runtime separates task behavior, model transport, tool execution,
speech input, and session storage.

## Turn lifecycle

1. Validate input and reserve `(session_id, request_id)` in SQLite.
2. Replay a saved result when the request and input match a completed turn.
3. Generate a plan from the task goal, context, and recent history.
4. Execute an action/observation loop within the tool and output-repair budgets.
5. Validate decisions and tool arguments against the registered schemas and allowlist.
6. Atomically commit the state version, history, and request result.

`ask` and `respond` end the current turn; `finish` also closes the session.
Applications define completion criteria. Planning runs each turn, with a goal-based fallback
for invalid plans. The turn deadline covers planning, model calls, and tools; ASR has a
separate deadline.

## Components

| Component | Responsibility |
| --- | --- |
| `ModelPort` | Async structured model responses |
| `TaskSpec` | Goals, instructions, allowed tools, and budgets |
| `ToolRegistry` | Schema validation and dispatch for async read-only tools |
| `Session` | Context, recent history, plan, version, and turn count |
| `SQLiteStore` | Request reservations, version checks, atomic commits, and replay |
| `VoiceInput` | Transcription followed by the shared runtime |
| `applications.py` | Interview and lexical notes-search applications |

## Concurrency and recovery

A unique partial index permits one running request per session. Model and tool calls run
between short database transactions. Commit and abort release reservations; version checks
detect conflicting updates. Completed results remain replayable after restart.

Stop the old worker before calling `recover` to release a crashed request. Async tool handlers
must support cancellation. ASR runs in a worker thread, so a timeout ends the wait while
inference may continue.

## Extension

Applications register typed async read-only tools and implement `ModelPort` for their model
backend. Session context remains fixed, while dialogue history is bounded by the task budget.
Task or tool-schema changes require a new session.

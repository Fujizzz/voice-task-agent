# Architecture and extension contracts

The runtime is an async Python library. There is no web server, global agent singleton,
framework-specific interview entity, or automatic microphone activation in the core.

## A turn

1. Validate input and reserve `(session_id, request_id)` in SQLite.
2. Return the saved result immediately if the same input was already committed.
3. Generate a structured plan from task goal, context and bounded history.
4. Execute a bounded action/observation loop. Each iteration consumes a tool call,
   output repair, or returns a final conversational action.
5. Validate model decisions and tool arguments. Only application-registered tools
   present in the task's allowlist may execute.
6. Atomically commit the new state version, history entry, and replayable result.

`ask` and `respond` both end the current turn and wait for another input. `finish`
also closes the session. Their meaning is application-controlled. The framework checks
the action schema, not the truth of an LLM's assertion that a business task is complete.
Applications with strict completion criteria must add deterministic domain validation.

## Component boundaries

- `ModelPort` is an async structured-output protocol; transport knows nothing about tools or interviews.
- `TaskSpec` supplies goals, instructions, allowed tools, and budgets.
- `ToolRegistry` describes and dispatches async read-only functions with typed arguments.
- `Session` contains context, bounded dialogue history, latest plan, version and turn count.
- `SQLiteStore` owns reservations, conflict checks, atomic publication, and request replay.
- `VoiceInput` transcribes before invoking the same runtime used for typed input.
- `applications.py` owns interviewer and knowledge-assistant behavior.

Planning is refreshed per turn. Invalid plans use a deterministic goal-based fallback;
invalid execution decisions use a limited shared repair counter. The overall turn deadline
includes planning, model rounds and tools. ASR has a separate input-stage deadline.

## Concurrency and recovery

A unique partial database index admits only one running request per session. The reservation
is released by commit or abort. No database transaction spans a network call. Version checks
protect publication even if another component modifies a session unexpectedly. A service
restart does not need to reconstruct an in-memory lock to replay completed turns.

Recovery is explicit because automatically expiring a lease could let two workers act at once.
The caller must stop the old worker before `recover`. Python cancellation cannot guarantee
termination of external servers or worker-thread inference. Third-party async tools must
cooperate with cancellation and must not block the event loop.

## Extending beyond read-only tasks

Add tools in an application, not in the execution loop. To support writes in a later version,
first design durable tool-call IDs, reservation/result records, ambiguity reconciliation after
timeouts, and application authorization. Turn-level idempotency alone cannot provide exactly-once
external execution. v0.1 rejects `read_only=False` tools instead of claiming this guarantee.

The current examples perform evidence lookup and conversational generation. They do not provide
production interview scoring, embeddings, a universal autonomous worker, or a hard guarantee
that prompt injection cannot influence model wording.

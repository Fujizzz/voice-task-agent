# Source provenance

Upstream: https://github.com/Fujizzz/AiInterviewer

Reviewed revision: `b21b331981d8a012a263ec1fc7a6e417f6f37082`

Refactor prepared on 2026-10-03. Commit dates record the actual refactor work, not an earlier
internship or a reconstruction of a historical implementation.

| Upstream source | Framework destination | Treatment |
| --- | --- | --- |
| `agents/ports/llm.py` | `ports.py` | Adapted the provider-neutral async structured-output protocol; prompt resolution belongs to the caller. |
| `agents/question/react.py` | `runtime.py` | Generalized the bounded loop invariant, repeated-call observations, schema repair, deadlines, and stop reasons. |
| `agents/planning/planner.py` | `runtime.py` / `contracts.py` | Replaced topic/difficulty/time allocations with a small task-neutral per-turn plan. |
| `agents/orchestrator/service.py` | `runtime.py` | Retained load/decide/commit separation and cancellation semantics; removed interview-specific branching. |
| `app/adapters/repository.py` and backend repository design | `storage.py` | Generalized version checks and replay; implemented a standalone SQLite reservation/commit backend. |
| `app/providers/llm.py` | `providers.py` | Reimplemented the transport boundary using HTTPX and compatible Chat Completions; no supplier-specific SDK types in core. |
| Agent integration tests | `tests/` | Reworked behavioral cases for a domain-neutral runtime, including loop limits, cancellation and replay. |

New framework-specific work: optional Whisper input, explicit microphone CLI, generic tool
registration, task contracts, standalone packaging, local-notes example, and cross-platform CI.

The interviewer example is a compact application of the extracted design. It is not the full
upstream product: scoring, candidate extraction, UI, recommendations, and other teams' modules
are not copied. Upstream contributors retain attribution for their work. The reviewed source
has no root license declaration, so this refactor does not assign a new blanket license to it.

# Validation scope

## Local verification on 2026-10-03

- Windows, Python 3.12.14: **47 tests passed**.
- Ruff lint and formatting checks passed for source, examples and tests.
- Built both a wheel and source distribution successfully.
- Real hosted/local LLM calls and physical microphone/Whisper inference were not run.
- Local test/build commands required execution outside the desktop filesystem sandbox
  because Windows denied access to tool-created temporary directories; no tests were skipped.

The core suite runs offline and does not send requests to a paid provider, download model
weights, or activate the microphone. Tests cover:

- Multiple applications using the same runtime and public CLI.
- Planning, tool observations, output repairs and bounded repeated-tool execution.
- Invalid/unauthorized tools, tool timeouts, context isolation and observation truncation.
- Shared deadlines, cancellation, failed commits and explicit crash recovery.
- Durable replay after reopening SQLite; request-ID conflicts and session configuration checks.
- Same-session concurrency across independent runtime/store instances.
- Hosted/local URL configuration and JSON modes through an injected HTTP transport.
- ASR-to-Agent data flow, empty recognition and ASR timeout using injected transcribers.

Run:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

## Environment acceptance

Before claiming a live voice deployment, run `--audio` with a known utterance and then
`--record` on the target microphone. Check the recognized text in the local session history.
Repeat with a real configured cloud API and a running local compatible server. Measure ASR
time, model time, total turn time and observed failures on representative inputs. Offline
tests are not evidence of real-model quality, latency, GPU compatibility or recognition accuracy.

This release provides no performance uplift numbers or production reliability claims.

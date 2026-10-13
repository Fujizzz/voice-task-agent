"""Text, audio-file, and explicit microphone entry points for the shared runtime."""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from .applications import application
from .providers import CompatibleModel, DemoModel
from .runtime import AgentRuntime
from .speech import VoiceInput, WhisperTranscriber, record_wav
from .storage import SQLiteStore


def parser():
    result = argparse.ArgumentParser(
        description="Voice-enabled task agents with planning and persistent sessions"
    )
    result.add_argument("--app", choices=["interviewer", "knowledge"], default="knowledge")
    result.add_argument("--demo", action="store_true", help="Use deterministic offline responses")
    result.add_argument(
        "--context", type=Path, help="UTF-8 JSON application context, required for new sessions"
    )
    result.add_argument("--db", default="sessions.sqlite3")
    result.add_argument("--session", help="Resume an existing session ID")
    result.add_argument("--request-id", help="Stable ID for idempotent retries of one text turn")
    inputs = result.add_mutually_exclusive_group()
    inputs.add_argument("--text")
    inputs.add_argument("--audio", type=Path)
    inputs.add_argument(
        "--record",
        type=float,
        metavar="SECONDS",
        help="Explicitly capture one microphone utterance",
    )
    result.add_argument(
        "--asr-model", default="base", help="Whisper model name or local model directory"
    )
    result.add_argument("--asr-device", default="cpu")
    result.add_argument("--asr-compute-type", default="int8")
    result.add_argument("--language", help="Optional ASR language code, e.g. zh or en")
    result.add_argument(
        "--recover",
        action="store_true",
        help="Release a crashed session reservation AFTER stopping its worker",
    )
    return result


def make_model(demo):
    if demo:
        return DemoModel()
    model = os.getenv("AGENT_MODEL", "")
    base_url = os.getenv("AGENT_BASE_URL", "")
    if not model or not base_url:
        raise ValueError("Set AGENT_MODEL and AGENT_BASE_URL, or use --demo")
    return CompatibleModel(
        base_url=base_url,
        model=model,
        api_key=os.getenv("AGENT_API_KEY", ""),
        response_format=os.getenv("AGENT_RESPONSE_FORMAT", "json_object"),
    )


async def run_cli(args):
    store = SQLiteStore(args.db)
    model = None
    try:
        if args.recover:
            if not args.session or args.text or args.audio or args.record is not None:
                raise ValueError("--recover requires --session and no input")
            print(json.dumps({"released_requests": store.recover(args.session)}))
            return
        if args.session and args.context:
            raise ValueError("Existing session context cannot be silently replaced")
        if args.request_id and args.text is None and args.audio is None and args.record is None:
            raise ValueError("--request-id is for a single input, not interactive chat")
        context = None
        if not args.session:
            if args.context is None:
                raise ValueError("New sessions require --context; see examples/*.json")
            context = json.loads(args.context.read_text(encoding="utf-8-sig"))
            if not isinstance(context, dict):
                raise ValueError("Context must be a JSON object")
            if args.app == "knowledge" and not isinstance(context.get("notes"), list):
                raise ValueError("Knowledge context requires a notes array")
        model = make_model(args.demo)
        task, tools = application(args.app)
        runtime = AgentRuntime(model, store, task, tools)
        session = args.session or runtime.create_session(context)
        print(f"session={session}", file=sys.stderr)
        if args.audio is not None or args.record is not None:
            voice = VoiceInput(
                runtime,
                WhisperTranscriber(
                    args.asr_model,
                    device=args.asr_device,
                    compute_type=args.asr_compute_type,
                    language=args.language,
                ),
            )
            if args.record is not None:
                with tempfile.TemporaryDirectory(prefix="voice-agent-") as directory:
                    path = str(Path(directory) / "utterance.wav")
                    print("Recording microphone input...", file=sys.stderr)
                    record_wav(path, seconds=args.record)
                    answer = await voice.run_file(session, path, request_id=args.request_id)
            else:
                answer = await voice.run_file(session, str(args.audio), request_id=args.request_id)
            print(answer.model_dump_json(indent=2))
        elif args.text is not None:
            answer = await runtime.run(session, args.text, request_id=args.request_id)
            print(answer.model_dump_json(indent=2))
        else:
            print("Enter text; /quit exits without a model call.", file=sys.stderr)
            while True:
                try:
                    text = input("> ")
                except EOFError:
                    break
                if text.strip() == "/quit":
                    break
                if not text.strip():
                    continue
                answer = await runtime.run(session, text)
                print(answer.text)
                if answer.action == "finish":
                    break
    finally:
        if isinstance(model, CompatibleModel):
            await model.aclose()
        store.close()


def main():
    args = parser().parse_args()
    try:
        asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        # Do not echo provider response bodies, keys or private context on an error path.
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

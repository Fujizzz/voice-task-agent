import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "voice_task_agent", *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )


@pytest.mark.parametrize(
    "app,text", [("knowledge", "What is Whisper?"), ("interviewer", "Start interview")]
)
def test_both_applications_run_through_public_cli_offline(tmp_path, app, text):
    process = invoke(
        "--demo",
        "--app",
        app,
        "--context",
        ROOT / "examples" / f"{app}.json",
        "--text",
        text,
        "--db",
        tmp_path / "sessions.sqlite",
        "--request-id",
        "one",
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["state_version"] == 1 and "Offline demo" in result["text"]
    assert any(event["component"] == "tool" for event in result["trace"])
    replay = invoke(
        "--demo",
        "--app",
        app,
        "--session",
        result["session_id"],
        "--text",
        text,
        "--db",
        tmp_path / "sessions.sqlite",
        "--request-id",
        "one",
    )
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout) == result


def test_third_application_needs_no_core_changes():
    process = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "custom_task.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["stop_reason"] == "COMPLETED"

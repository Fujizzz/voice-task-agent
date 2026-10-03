import asyncio
import json
import sys
from types import SimpleNamespace

import httpx
import pytest
from conftest import ScriptedModel

from voice_task_agent.contracts import Plan
from voice_task_agent.errors import InvalidModelOutput, ModelError
from voice_task_agent.providers import CompatibleModel
from voice_task_agent.speech import VoiceInput, WhisperTranscriber, record_wav


@pytest.mark.parametrize("base_url", ["https://hosted.example/v1", "http://127.0.0.1:8000/v1"])
@pytest.mark.parametrize("format", ["json_object", "json_schema", "text"])
async def test_same_adapter_supports_hosted_and_local_protocols(base_url, format):
    def server(request):
        assert str(request.url) == base_url + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-placeholder"
        body = json.loads(request.content)
        assert body["model"] == "configured-model"
        assert ("response_format" in body) == (format != "text")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"steps":["Inspect context"]}'},
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        model = CompatibleModel(
            base_url=base_url,
            model="configured-model",
            api_key="test-placeholder",
            response_format=format,
            client=client,
        )
        result = await model.generate_structured(prompt="plan", payload={}, schema=Plan)
        assert result.steps == ["Inspect context"]


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": '{"steps":[]}'}}]},
    ],
)
async def test_bad_provider_payload_is_not_accepted(response):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        model = CompatibleModel(base_url="http://localhost/v1", model="test", client=client)
        with pytest.raises(InvalidModelOutput):
            await model.generate_structured(prompt="plan", payload={}, schema=Plan)


async def test_http_errors_do_not_expose_secret_response_or_retry():
    calls = []

    def server(request):
        calls.append(1)
        return httpx.Response(401, text="secret-response-content")

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        model = CompatibleModel(base_url="http://localhost/v1", model="test", client=client)
        with pytest.raises(ModelError) as error:
            await model.generate_structured(prompt="plan", payload={}, schema=Plan)
    assert "secret-response-content" not in str(error.value)
    assert len(calls) == 1


async def test_transcribed_text_enters_exact_same_runtime_and_persists(factory):
    class ASR:
        async def transcribe(self, path):
            assert path == "sample.wav"
            return "  Explain task planning  "

    runtime, store, _ = factory(ScriptedModel())
    session = runtime.create_session()
    result = await VoiceInput(runtime, ASR()).run_file(
        session, "sample.wav", request_id="voice-one"
    )
    assert result.request_id == "voice-one"
    assert store.get(session).history[0].user == "Explain task planning"


async def test_empty_asr_does_not_create_agent_turn(factory):
    class ASR:
        async def transcribe(self, path):
            return " "

    runtime, store, model = factory()
    session = runtime.create_session()
    with pytest.raises(ValueError):
        await VoiceInput(runtime, ASR()).run_file(session, "silent.wav")
    assert model.calls == [] and store.get(session).version == 0


async def test_asr_timeout_does_not_reserve_session(factory):
    class ASR:
        async def transcribe(self, path):
            await asyncio.sleep(10)

    runtime, store, model = factory()
    session = runtime.create_session()
    with pytest.raises(TimeoutError):
        await VoiceInput(runtime, ASR(), asr_seconds=0.01).run_file(session, "sample.wav")
    assert model.calls == [] and store.get(session).version == 0


async def test_whisper_lazy_loading_and_missing_file():
    adapter = WhisperTranscriber()
    assert adapter._model is None
    with pytest.raises(FileNotFoundError):
        await adapter.transcribe("nonexistent-test-input.wav")


@pytest.mark.parametrize("seconds", [-1, 0, 121, float("inf"), float("nan")])
def test_invalid_recording_duration_rejected_before_audio_import(seconds):
    with pytest.raises(ValueError):
        record_wav("unused.wav", seconds=seconds)


async def test_whisper_adapter_consumes_lazy_segments_and_reuses_model(tmp_path, monkeypatch):
    loads = []
    calls = []

    class FakeWhisper:
        def __init__(self, name, **kwargs):
            loads.append((name, kwargs))

        def transcribe(self, path, **kwargs):
            calls.append((path, kwargs))
            return iter([SimpleNamespace(text=" Hello "), SimpleNamespace(text="world")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisper))
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"adapter-test-placeholder")
    adapter = WhisperTranscriber("local-model-directory", language="en")
    assert await adapter.transcribe(str(audio)) == "Hello world"
    assert await adapter.transcribe(str(audio)) == "Hello world"
    assert len(loads) == 1 and len(calls) == 2
    assert calls[0][1] == {"language": "en", "vad_filter": True}


def test_microphone_adapter_writes_valid_pcm_wav(tmp_path, monkeypatch):
    import wave

    stopped = []

    class PCM:
        def astype(self, dtype):
            assert dtype == "<i2"
            return self

        def tobytes(self):
            return b"\x00\x00" * 16000

    def record(frames, **kwargs):
        assert frames == 16000
        assert kwargs == {"samplerate": 16000, "channels": 1, "dtype": "int16"}
        return PCM()

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(rec=record, wait=lambda: None, stop=lambda: stopped.append(True)),
    )
    path = tmp_path / "recording.wav"
    record_wav(str(path), seconds=1)
    with wave.open(str(path), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 16000
        assert reader.getnframes() == 16000
    assert stopped == [True]

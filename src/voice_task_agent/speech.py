"""Audio-segment transcription and microphone recording."""

import asyncio
import math
import threading
from pathlib import Path

from .ports import TranscriberPort


class WhisperTranscriber:
    """Local Whisper transcription with lazy loading from a model name or directory."""

    def __init__(
        self,
        model: str = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ):
        self.model_name, self.device = model, device
        self.compute_type, self.language = compute_type, language
        self._model = None
        self._lock = threading.Lock()

    def _transcribe(self, path):
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    raise RuntimeError("Install voice-task-agent[voice] to use Whisper") from None
                self._model = WhisperModel(
                    self.model_name, device=self.device, compute_type=self.compute_type
                )
            segments, _ = self._model.transcribe(path, language=self.language, vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe(self, path: str) -> str:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return await asyncio.to_thread(self._transcribe, path)


def record_wav(path: str, *, seconds: float = 5, sample_rate: int = 16000):
    """Record a fixed-duration audio segment."""
    import wave

    if not math.isfinite(seconds) or not 0 < seconds <= 120 or sample_rate != 16000:
        raise ValueError("Recording requires 0 < seconds <= 120 and sample_rate=16000")
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("Install voice-task-agent[voice] for microphone capture") from None
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    try:
        sd.wait()
    finally:
        sd.stop()
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio.astype("<i2").tobytes())


class VoiceInput:
    def __init__(self, runtime, transcriber: TranscriberPort, *, asr_seconds: float = 120):
        if not math.isfinite(asr_seconds) or asr_seconds <= 0:
            raise ValueError("ASR timeout must be positive and finite")
        self.runtime, self.transcriber, self.asr_seconds = runtime, transcriber, asr_seconds

    async def run_file(self, session_id: str, path: str, *, request_id: str | None = None):
        async with asyncio.timeout(self.asr_seconds):
            text = await self.transcriber.transcribe(path)
        if not text.strip():
            raise ValueError("No speech was recognized; no agent turn was created")
        return await self.runtime.run(session_id, text, request_id=request_id)

"""DTLN noise suppression for the candidate's microphone.

Ported from voice-bot-poc (pipecat_extend/audio/filters/dtln_filter.py) to a LiveKit
FrameProcessor, so it runs on the room's audio stream before VAD, turn detection and STT.
DTLN: https://github.com/breizhn/DTLN (MIT). Models live in ai/models/dtln.
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime
from livekit import rtc

log = logging.getLogger("hiremind.dtln")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "dtln"
SAMPLE_RATE = 16000  # DTLN only runs at 16 kHz; the room input is configured to match
BLOCK_LEN = 512  # 32 ms
BLOCK_SHIFT = 128  # 8 ms

_sessions: tuple[onnxruntime.InferenceSession, onnxruntime.InferenceSession] | None = None


def _load() -> tuple[onnxruntime.InferenceSession, onnxruntime.InferenceSession]:
    """ONNX sessions are shared by every call in the worker; per-call state lives in DTLNFilter."""
    global _sessions
    if _sessions is None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _sessions = (
            onnxruntime.InferenceSession(str(MODEL_DIR / "model_1.onnx"), sess_options=opts),
            onnxruntime.InferenceSession(str(MODEL_DIR / "model_2.onnx"), sess_options=opts),
        )
    return _sessions


def _zeros(session: onnxruntime.InferenceSession) -> dict[str, np.ndarray]:
    return {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], dtype=np.float32) for i in session.get_inputs()}


class DTLNFilter(rtc.FrameProcessor[rtc.AudioFrame]):
    """Streams 16 kHz mono frames through DTLN. Output frames have the input's length (8 ms added latency)."""

    def __init__(self) -> None:
        self._m1, self._m2 = _load()
        self._names1 = [i.name for i in self._m1.get_inputs()]
        self._names2 = [i.name for i in self._m2.get_inputs()]
        self._enabled = True
        self.reset()

    def reset(self) -> None:
        self._in1, self._in2 = _zeros(self._m1), _zeros(self._m2)
        self._in_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
        self._out_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        # One block of silence up front means a full frame of output is always ready.
        self._ready = np.zeros(BLOCK_SHIFT, dtype=np.float32)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _block(self) -> np.ndarray:
        spec = np.fft.rfft(self._in_buf)
        mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
        self._in1[self._names1[0]] = mag
        mask, self._in1[self._names1[1]] = self._m1.run(None, self._in1)
        est = np.fft.irfft(mag * mask * np.exp(1j * np.angle(spec))).reshape(1, 1, -1).astype(np.float32)
        self._in2[self._names2[0]] = est
        out, self._in2[self._names2[1]] = self._m2.run(None, self._in2)
        return np.squeeze(out)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1:
            return frame
        audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, audio])
        produced = []
        while len(self._pending) >= BLOCK_SHIFT:
            self._in_buf = np.roll(self._in_buf, -BLOCK_SHIFT)
            self._in_buf[-BLOCK_SHIFT:] = self._pending[:BLOCK_SHIFT]
            self._pending = self._pending[BLOCK_SHIFT:]
            self._out_buf += self._block()
            produced.append(self._out_buf[:BLOCK_SHIFT].copy())
            self._out_buf = np.roll(self._out_buf, -BLOCK_SHIFT)
            self._out_buf[-BLOCK_SHIFT:] = 0
        if produced:
            self._ready = np.concatenate([self._ready, *produced])
        n = frame.samples_per_channel
        out, self._ready = self._ready[:n], self._ready[n:]
        pcm = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
        return rtc.AudioFrame(pcm.tobytes(), SAMPLE_RATE, 1, n)

    def _close(self) -> None:
        # The room re-subscribes when the candidate's track changes; start the next stream clean.
        self.reset()

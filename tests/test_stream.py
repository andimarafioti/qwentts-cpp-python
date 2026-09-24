"""Packet boundaries, native buffer ownership, and cooperative cancellation."""
import ctypes
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from qwentts_cpp import QwenTTS, QwenTTSError
from qwentts_cpp._binding import CODEC_FRAME_SAMPLES, QT_ABI_VERSION, QwenStatus


class NativeStream:
    def __init__(self, frames=39, status=QwenStatus.OK, wait_for_cancel=False):
        self.audio = np.arange(frames * CODEC_FRAME_SAMPLES, dtype=np.float32)
        self.status = status
        self.wait_for_cancel = wait_for_cancel
        self.cancelled = threading.Event()
        self.freed = threading.Event()
        self.called = False

    def qt_tts_default_params(self, params):
        params._obj.abi_version = QT_ABI_VERSION

    def qt_synthesize(self, ctx, params, audio):
        self.called = True
        params = params._obj
        offset, width = 0, 1
        while offset < self.audio.size:
            if params.cancel(None):
                self.cancelled.set()
                return QwenStatus.CANCELLED
            end = min(offset + width * CODEC_FRAME_SAMPLES, self.audio.size)
            buffer = self.audio[offset:end].copy()
            proceed = params.on_chunk(
                buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buffer.size, None,
            )
            buffer.fill(-1)  # Native callback memory is no longer valid.
            if not proceed:
                self.cancelled.set()
                return QwenStatus.CANCELLED
            offset = end
            width = min(width * 2, 8)
        if self.wait_for_cancel:
            # The consumer receives a packet, closes the generator, and the
            # native cancellation callback must observe that promptly.
            for _ in range(500):
                if params.cancel(None):
                    # Even if native delivers an in-flight chunk after close,
                    # the callback must reject it without queueing more PCM.
                    assert not params.on_chunk(
                        buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buffer.size, None,
                    )
                    self.cancelled.set()
                    return QwenStatus.CANCELLED
                self.cancelled.wait(0.01)
            raise AssertionError("Stream close did not cancel native synthesis")
        return self.status

    def qt_audio_free(self, audio):
        self.freed.set()


def make_tts(native):
    tts = QwenTTS.__new__(QwenTTS)
    tts._ctx = 123
    tts._lock = threading.Lock()
    tts.library = SimpleNamespace(_lib=native, last_error=lambda: "native failure")
    return tts


@pytest.mark.parametrize("first", [1, 2, 4, 8])
@pytest.mark.parametrize("seconds,later", [(0.01, 1), (0.32, 4), (0.64, 8), (1.0, 13)])
def test_packet_sizes_and_audio_are_independent_of_native_ramp(first, seconds, later):
    native = NativeStream()
    tts = make_tts(native)
    packets = list(tts.stream(text="test", first_chunk_frames=first, codec_chunk_sec=seconds))
    sizes = [packet.size // CODEC_FRAME_SAMPLES for packet, _ in packets]
    full, tail = divmod(39 - first, later)
    assert sizes == [first] + [later] * full + ([tail] if tail else [])
    assert all(rate == 24000 for _, rate in packets)
    np.testing.assert_array_equal(np.concatenate([p for p, _ in packets]), native.audio)
    assert native.freed.is_set()
    profile = tts.last_stream_profile
    assert profile["first_callback_n_samples"] == CODEC_FRAME_SAMPLES
    assert profile["first_packet_n_samples"] == first * CODEC_FRAME_SAMPLES
    assert profile["first_callback_enter_ms"] <= profile["first_packet_ready_ms"] <= profile["first_yield_ms"]
    assert profile["packet_count"] == len(packets)


def test_default_packet_sequence_is_four_then_eight_frames_and_short_tail():
    packets = list(make_tts(NativeStream()).stream(text="test"))
    assert [packet.size for packet, _ in packets] == [
        frames * CODEC_FRAME_SAMPLES for frames in (4, 8, 8, 8, 8, 3)
    ]


@pytest.mark.parametrize("frames", [0, 1, 3, 7])
def test_short_utterance_flushes_at_successful_eos(frames):
    native = NativeStream(frames=frames)
    packets = list(make_tts(native).stream(text="test", first_chunk_frames=8))
    assert len(packets) == (1 if frames else 0)
    if frames:
        np.testing.assert_array_equal(packets[0][0], native.audio)
    assert native.freed.is_set()


@pytest.mark.parametrize("value", [0, -1, 3, 16, 4.0, True, "4", None])
def test_invalid_first_packet_fails_before_native_call(value):
    native = NativeStream()
    with pytest.raises(ValueError, match="first_chunk_frames"):
        list(make_tts(native).stream(text="test", first_chunk_frames=value))
    assert not native.called


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_later_packet_fails_before_native_call(seconds):
    native = NativeStream()
    with pytest.raises(ValueError, match="codec_chunk_sec"):
        list(make_tts(native).stream(text="test", codec_chunk_sec=seconds))
    assert not native.called


@pytest.mark.parametrize("status", [QwenStatus.GENERATE_FAILED, QwenStatus.CANCELLED])
def test_failure_discards_unfinished_packet_and_frees_native_audio(status):
    native = NativeStream(frames=3, status=status)
    stream = make_tts(native).stream(text="test", first_chunk_frames=4)
    with pytest.raises(QwenTTSError, match="native failure"):
        next(stream)
    assert native.freed.is_set()


def test_failure_after_first_packet_preserves_packet_then_raises():
    native = NativeStream(frames=7, status=QwenStatus.GENERATE_FAILED)
    stream = make_tts(native).stream(text="test", first_chunk_frames=4)
    np.testing.assert_array_equal(next(stream)[0], native.audio[:4 * CODEC_FRAME_SAMPLES])
    with pytest.raises(QwenTTSError, match="native failure"):
        next(stream)
    assert native.freed.is_set()


def test_repeated_stream_resets_packet_state():
    tts = make_tts(NativeStream(frames=3))
    for _ in range(2):
        packets = list(tts.stream(text="test", first_chunk_frames=4))
        assert len(packets) == 1
        assert packets[0][0].size == 3 * CODEC_FRAME_SAMPLES
        assert tts.last_stream_profile["packet_count"] == 1


def test_close_cancels_native_generation_with_partial_packet_pending():
    native = NativeStream(frames=7, wait_for_cancel=True)
    tts = make_tts(native)
    stream = tts.stream(text="test", first_chunk_frames=4)
    assert next(stream)[0].size == 4 * CODEC_FRAME_SAMPLES
    stream.close()
    assert native.cancelled.is_set()
    assert native.freed.is_set()
    assert tts.last_stream_profile["packet_count"] == 1
    assert not tts._lock.locked()

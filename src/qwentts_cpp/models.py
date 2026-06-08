from __future__ import annotations

from pathlib import Path
from typing import Tuple

GGUF_REPO = "Serveurperso/Qwen3-TTS-GGUF"

_MODEL_TO_TALKER_STEM = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "qwen-talker-0.6b-base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "qwen-talker-0.6b-customvoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "qwen-talker-1.7b-base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "qwen-talker-1.7b-customvoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "qwen-talker-1.7b-voicedesign",
}


def _normalize_quant(quant: str) -> str:
    normalized = quant.upper()
    aliases = {
        "F32": "F32",
        "FP32": "F32",
        "BF16": "BF16",
        "Q8": "Q8_0",
        "Q8_0": "Q8_0",
        "Q4": "Q4_K_M",
        "Q4_K_M": "Q4_K_M",
    }
    if normalized not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported qwentts quant {quant!r}. Expected one of: {allowed}")
    return aliases[normalized]


def resolve_gguf_paths(
    model_id: str,
    *,
    quant: str = "BF16",
    repo_id: str = GGUF_REPO,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Tuple[Path, Path]:
    """Resolve a Qwen3-TTS HF model id to qwentts.cpp talker/codec GGUF paths."""
    if model_id not in _MODEL_TO_TALKER_STEM:
        known = ", ".join(sorted(_MODEL_TO_TALKER_STEM))
        raise ValueError(f"Unsupported Qwen3-TTS model id {model_id!r}. Known ids: {known}")

    from huggingface_hub import hf_hub_download

    q = _normalize_quant(quant)
    talker_name = f"{_MODEL_TO_TALKER_STEM[model_id]}-{q}.gguf"
    codec_name = f"qwen-tokenizer-12hz-{q}.gguf"

    kwargs = {
        "repo_id": repo_id,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    talker = Path(hf_hub_download(filename=talker_name, **kwargs))
    codec = Path(hf_hub_download(filename=codec_name, **kwargs))
    return talker, codec

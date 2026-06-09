# PR wheel audio samples

These samples were generated on the local GB10 machine, using cached GGUF
files and no model downloads. The CPU files were regenerated from a local CPU
wheel built at native ref `a62fde62e64bf81a49af6e6c07ac5817df93f19e`; the
newer `eda8b59092b8d7b7142177c599b08909b87f63b4` CPU output failed STT on
aarch64. CUDA samples use the newer ref.

Common settings:

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Quantization: `BF16`
- Speaker: `aiden`
- Language: `english`
- Seed: `42`
- Sampling settings: `temperature=0.9`, `top_k=50`, `top_p=1.0`,
  `repetition_penalty=1.05`
- Streaming chunk: `8 / 12.5` seconds, with `2.0` seconds left context

Files:

| File | Mode | Prompt | Duration |
| --- | --- | --- | --- |
| `cpu_greedy_20f_customvoice_bf16.wav` | Greedy, 20 frames | `This is a short deterministic backend comparison.` | 1.60s |
| `cu128_greedy_20f_customvoice_bf16.wav` | Greedy, 20 frames | `This is a short deterministic backend comparison.` | 1.60s |
| `cu130_greedy_20f_customvoice_bf16.wav` | Greedy, 20 frames | `This is a short deterministic backend comparison.` | 1.60s |
| `cpu_sampled_seed42_customvoice_bf16.wav` | Sampled, seed 42 | `Hello from the qwentts cpp Python wheel. This sample compares the same Qwen three custom voice settings across backends.` | 6.40s |
| `cu128_sampled_seed42_customvoice_bf16.wav` | Sampled, seed 42 | `Hello from the qwentts cpp Python wheel. This sample compares the same Qwen three custom voice settings across backends.` | 6.40s |
| `cu130_sampled_seed42_customvoice_bf16.wav` | Sampled, seed 42 | `Hello from the qwentts cpp Python wheel. This sample compares the same Qwen three custom voice settings across backends.` | 6.40s |

Validation:

- STT validator: `nano-parakeet` with `nvidia/parakeet-tdt-0.6b-v3`.
- All six final WAVs produced non-empty, prompt-matching transcripts.

See `manifest.json` for the measured generation metadata and STT transcripts.

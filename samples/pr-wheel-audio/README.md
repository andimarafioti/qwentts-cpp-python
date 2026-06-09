# PR wheel audio samples

These samples were generated from the PR wheel artifacts on the local GB10
machine, using cached GGUF files and no model downloads.

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
| `cpu_sampled_seed42_customvoice_bf16.wav` | Sampled, seed 42 | `Hello from the qwentts cpp Python wheel. This sample compares the same Qwen three custom voice settings across backends.` | 3.68s |
| `cu128_sampled_seed42_customvoice_bf16.wav` | Sampled, seed 42 | `Hello from the qwentts cpp Python wheel. This sample compares the same Qwen three custom voice settings across backends.` | 6.40s |
| `cu130_sampled_seed42_customvoice_bf16.wav` | Sampled, seed 42 | `Hello from the qwentts cpp Python wheel. This sample compares the same Qwen three custom voice settings across backends.` | 6.40s |

See `manifest.json` for the measured generation metadata.

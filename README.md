# Anima Lightning for Forge Neo

[한국어 설명](README.ko.md)

Run [aina-tech/Anima-Lightning](https://huggingface.co/aina-tech/Anima-Lightning) in
[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) while retaining its native prompt, attention, LoRA and sampler paths.
No Forge source files or other extensions are modified.

## Features

- Preserves the model's learned TDM-Unify **K-conditioning branch**.
- K follows the UI's sampling steps, including separate Hires steps. Other samplers such as Euler also receive the matching K condition.
- Adds **Anima Lightning**, an Euler sampler with the official shift-3 schedule, extended to the selected step count.
- At 4 steps, the sigma sequence is exactly `[1, 0.9, 0.75, 0.5, 0]`.
- No preset checkbox, accordion, or automatic override of steps, CFG or sampler selection.
- Detects supported models by their K-branch tensor keys, not their filename.

The official starting recipe is **4 steps, CFG 1, shift 3**, at 1024×1024.
Increasing steps is supported, but does not guarantee better image quality.

## Install

Use Forge Neo's **Extensions → Install from URL** with:

```text
https://github.com/david419kr/sd-forge-anima-lightning
```

Or run from the Forge Neo directory:

```powershell
git clone https://github.com/david419kr/sd-forge-anima-lightning extensions/sd-forge-anima-lightning
```

Restart Forge after installation. No additional Python packages are installed automatically.
This extension requires a working **native Anima setup in Forge Neo**, including its
Qwen3 0.6B text encoder and compatible VAE. Reuse those components when selecting the converted checkpoint.

## Prepare the checkpoint

**Model weights are not included.** From the Forge Neo directory, use its existing Python environment.
The following Windows commands assume `venv/Scripts/python.exe` and `venv/Scripts/hf.exe`;
adjust those paths for your installation.

Download the supported source revision from the model publisher:

```powershell
.\venv\Scripts\hf.exe download aina-tech/Anima-Lightning `
  --revision 9dbb6e1a0f94eda0a1e523836d32355fb961cc7a `
  --local-dir extensions/sd-forge-anima-lightning/cache/source `
  --include "transformer/*" "text_conditioner/*" "LICENSE.md" "NOTICE.md"
```

Convert it once:

```powershell
.\venv\Scripts\python.exe extensions/sd-forge-anima-lightning/tools/convert_model.py `
  extensions/sd-forge-anima-lightning/cache/source `
  models/Stable-diffusion/Anima-Lightning-K4-shift3-forge.safetensors
```

The converter preserves all learned tensors, rejects unknown transformer keys,
verifies the source transformer SHA-256, and refuses to overwrite an existing output.
It does not execute downloaded Python code.
The legacy `K4` filename does **not** limit the current K value; renaming the file is also fine.

Select the converted checkpoint and your usual Anima text encoder/VAE in Forge.
Choose **Anima Lightning** and set your own sampling steps and CFG.

## Sampling behavior

| Setting | Behavior |
|---|---|
| Anima Lightning sampler | Euler, deterministic churn defaults, fixed shift-3 sigma schedule with the requested number of steps |
| Schedule type / UI Shift with that sampler | Do not change its fixed shift-3 sigma schedule |
| Other samplers | Retain their normal schedule, UI Shift and CFG behavior |
| K condition | `3 / (K + 2)`, using the model's trained conditioning shift of 3 |
| Base pass | K follows `Steps` |
| Hires pass | K follows `Hires steps`, falling back to base steps when it is 0 |
| Img2img / Hires denoise strength | Forge's native schedule expansion/truncation remains; K follows that pass's UI step count |

The old sampler name `Anima Lightning (4 steps)` is accepted as an infotext/API alias.
At CFG 1, ordinary negative-prompt CFG has no effect; NegPiP's inline negative weights use a separate attention path.
No extra controls are added to the generation UI. Infotext records the K used for each pass.

## Compatibility and validation

Tested in an existing Forge Neo installation at `6009ffff`, with PyTorch 2.11.0+cu130,
Diffusers 0.37.1 and an RTX 4090:

- FP32/BF16 K=4/6/8 embeddings compared against Diffusers.
- Every converted tensor key and shape checked against the native model and text adapter.
- The original 4-step output remained pixel-identical after variable-K support was added.
- Actual generation at 4/6/8 steps, including 4→6→8→4 state reset.
- NegPiP, NegPiP + Forge Couple, and NegPiP + LoRA stage scheduling.
- Separate Hires K values, img2img, ordinary Euler settings, and switching back to regular Anima.
- No preset UI; legacy sampler-name resolution.

Run the local checks from the Forge directory:

```powershell
.\venv\Scripts\python.exe extensions/sd-forge-anima-lightning/tests/test_support.py
```

The tensor inventory check is skipped if the converted checkpoint is absent.
The installed NegPiP wrapper is also exercised when its Anima hook is available.
Integration logs, images, downloaded files and installation-specific harnesses are excluded from this repository.

All combinations of extensions, ADetailer/ControlNet execution, `torch.compile`, replacement
text encoders, and future Forge versions have not been validated.
The extension uses Forge internals, so upstream loader changes may require an update.

## License and credits

Extension source: [GNU AGPL v3](LICENSE). See [NOTICE.md](NOTICE.md) for upstream acknowledgements.

**The model has a separate license.** Anima-Lightning remains subject to the
[CircleStone Labs non-commercial license](https://huggingface.co/aina-tech/Anima-Lightning/blob/main/LICENSE.md)
and its upstream terms. This repository does not distribute model weights or change their licensing.

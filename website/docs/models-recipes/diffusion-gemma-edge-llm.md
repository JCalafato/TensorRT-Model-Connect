---
title: DiffusionGemma block diffusion
---

# DiffusionGemma block diffusion

The `diffusion_gemma` family owns the prepared
`nvidia/diffusiongemma-26B-A4B-it-NVFP4` checkpoint for text and ordered
image/text requests. Its native fallback uses TensorRT Python graph construction
and Model Connect's C++ runtime, without ONNX or custom GPU kernels.

```sh
python -m tensorrt_model_connect build /path/to/diffusiongemma-26B-A4B-it-NVFP4 \
  --family diffusion_gemma --task images_text_to_text \
  --precision fp16 --quantization nvfp4 --max-sequence-length 4096 \
  -o diffusiongemma.bundle
```

Optional Edge 0.11.0 tools may be installed from a wheel or source. The pinned
Edge DiffusionGemma graph has a reproduced undefined-`token_phase` error, so this
family warns and retains the full native request instead of advertising a
working Edge engine. Native construction preserves prepared expert quantization
and does not start a new calibration campaign.

The public runtime exposes `TextContinuation` and `ImagesTextToText`. Diffusion
canvas admission, image order and features, sampler controls, and final-answer
rendering are family-owned. The whole final canvas must fit the bundle capacity.

The full native build succeeded, and fresh public C++ text, single-image and
two-image execution passed all six frozen known-answer checks. An independent
BF16 vLLM reference also passed those six answers. FP16 component discrepancies
remain recorded separately; these results do not imply equal-precision logit
parity or broad precision/platform qualification. The approved eager-mask and
same-image reference corrections are applied without tolerance changes. The
recorded full-model runs predate the latest text and vision parameter-rounding
fixes; a fresh full-checkpoint build of that corrected graph is not claimed.
See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/diffusion_gemma/README.md)
for the exact checkpoint, supported input contracts, known failures and validation
boundaries.

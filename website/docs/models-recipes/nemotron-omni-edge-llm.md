---
title: Nemotron Omni
---

# Nemotron Omni

The `nemotron_omni` family owns the officially documented prepared
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` checkpoint. It implements
text, image, acoustic and temporal-video reasoning using native TensorRT Python
graph construction and the Model Connect C++ runtime, without ONNX.

Optional Edge-LLM 0.11.0 offload is family-owned. Compatible source or wheel
Python tools use the existing CMake native SDK discovery contract. Text/image
tasks may use Edge; known audio/video execution failures select the native
implementation instead. Failed Edge preparation warns and retries native once.

```sh
python -m tensorrt_model_connect build /path/to/Nemotron-Omni \
  --family nemotron_omni --task vision_language_generation \
  --precision fp16 --quantization nvfp4 --max-sequence-length 4096 \
  -o nemotron-omni.bundle
```

## Evidence boundaries

The full native build and all six public task executions succeed. Text, audio
and audio/video pass the unchanged independent quality gate; image, video and
image/audio have recorded accuracy failures and remain runnable. The public
Model Connect Edge build and text/image executions also succeed: text passes
quality, while image fails the unchanged gate and remains runnable. Small
component tests alone are not full-model evidence. Direct official Edge has the
same text-pass/image-accuracy-failure classification, without establishing a
shared cause. Direct Edge
audio and typed-video execution errors are recorded separately.

Small native FP8 and NVFP4 projections execute correctly on Ada; lack of FP4
acceleration alone is not a correctness restriction. TensorRT rejects these
FP8-containing graphs on A30. These component observations are not full-model
qualification or a family-owned GPU allowlist.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_omni/README.md)
for pinned checkpoint details, public task bindings, capacity/timing contracts,
installation behavior and reproducible owning validation.

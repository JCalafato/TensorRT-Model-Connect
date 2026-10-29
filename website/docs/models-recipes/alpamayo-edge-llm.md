---
title: Alpamayo driving reasoning
---

# Alpamayo driving reasoning

The `alpamayo` family implements the documented FP16 `nvidia/Alpamayo-R1-10B`
configuration, with sample capacities one and six. Four camera sequences and ego
history produce reasoning and future physical trajectories through Model
Connect's existing structured-decision Task interface.

```sh
python -m tensorrt_model_connect alpamayo build /path/to/Alpamayo-R1-10B \
  --execution-variant native --max-batch-size 6 \
  --max-sequence-length 4096 -o alpamayo.bundle
```

Native execution uses TensorRT Python graph construction and the Model Connect
C++ runtime, without ONNX. The default `auto` path can use compatible installed
Edge-LLM 0.11.0 wheel/source tools and SDK. Failed Edge preparation warns and
retries native once; publication or inference failures do not change backends.
Selection and adapters are entirely family-owned.

## Evidence boundaries

Actual native B1/B6 build and raw-media inference execute. One public B6 clip
measured minADE6 at 6.4 seconds of 0.7304 m against a 0.90 m limit. This does not
reproduce the broader evaluation. Separate FP16 text/vision numerical
gates remain failing and are not waived by that trajectory result.

The tested x86 Edge B1/B6 commands also execute, but the B6 trajectory result of
3.0877 m fails the same limit and reasoning is incorrect. Accuracy-only failures
remain runnable. A tested ARM source-SDK compiler failure is recorded separately.
These limitations prevent a clean qualification claim; they are not a native
GPU allowlist or evidence of all-platform support.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/alpamayo/README.md)
for exact checkpoint, task contracts, commands and validation scope.

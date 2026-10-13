# Gemma4 Edge-LLM execution

Gemma owns complete-network offload to the official Edge-LLM 0.11.0 snapshot,
revision `95515c2f87fba8982db5a519f9022277667b3cc9`. The family maps the upstream
ONNX exporter, native component builders and persistent C++ runtime. No shared
model selection, cross-compilation or model-time installation is introduced.
Earlier Gemma generations keep their existing native implementation.

## Installation and build

Provision the [native SDK](../../../cmake/edge_llm/README.md) with
`TRTMC_EDGELLM_ALL_KERNELS=ON` and `TRTMC_EDGELLM_ONNX=ON`, configure Model
Connect with `TRTMC_ENABLE_EDGELLM=ON`, and expose the installation through
`CMAKE_PREFIX_PATH`. The 0.11 SDK prerequisite also installs the upstream
`tensorrt-edgellm[export]` Python dependencies. Pip tools do not replace the
C++ headers, archives, plugin and native builders required by this adapter.

```sh
# Standalone: export and build the checkpoint's text and available media towers.
trtmc gemma build /path/to/gemma-4-E2B-it --precision fp16 -o gemma.bundle

# Explicit text-only target/draft pair.
trtmc gemma build /path/to/gemma-4-12B-it --precision fp16 \
  --execution-variant mtp --companion draft=/path/to/gemma-4-12B-it-assistant \
  -o gemma-mtp.bundle
```

The family-owned `--execution-variant` values are `mtp`, `dspark`, `eagle3` and
`dflash`. The draft must be an existing local checkpoint with compatible
geometry; it is never inferred or downloaded. Options use the existing family
CLI protocol from #1310, not new shared flags or hooks. Python callers use
`GemmaBuildRequest`, `BuildExecutionInputs` and `NamedCheckpoint` from this
family's `edge_llm.config` with the unchanged public build API.

Standalone builds use FP16 compute, TP1 and batch1. Original checkpoints retain
their unquantized weights. The published 26B-A4B ModelOpt NVFP4 checkpoint also
keeps its packed weights and FP8 KV cache on native SM12x; no calibration or
quantization conversion is performed. They expose text
continuation and the image/audio tasks supported by the checkpoint's towers.
The runtime preserves ordered typed media and delegates preprocessing to Edge;
images are host RGB8 or normalized RGB float, and audio is mono 16 kHz float
PCM. Unsupported content, generation options and unmapped build overrides are
rejected rather than silently discarded. Paired builds are text-only and mapped
to native Linux x86_64 SM80. Other packed layouts and quantized companion pairs are not admitted by
this change. Each standalone build resolves only its actual media towers, using
the installed SDK's declared visual/audio executable paths.

The builder packages validated engines, tokenizer/chat-template files and
required ordinary/per-layer embeddings. It does not duplicate original weights
or ONNX intermediates into the bundle. Edge 0.11 processes the exported
`chat_template.jinja`; the old 0.10 static-template workaround is not carried
forward. Output comparison does not strip generated text or relax gates.

Ordinary Edge preparation failures retain a diagnostic log, warn and attempt
the existing native builder. Gemma4 has no equivalent native implementation, so
failure remains explicit. A failed companion request never produces a base-only
bundle. Publication and runtime failures propagate without backend switching.
Use scratch-backed output and `TMPDIR` with enough space for export and bundle
extraction; these operations are bounded-memory, not zero-disk-space.

## Recorded 0.11 validation

These are prior local Model Connect build/inference results using native CUDA
13.3 / TensorRT 11.1.0.106, original weights, FP16, TP1/batch1 and the existing
family validation helpers. They are not new-PR-head CI or full-family approval.
No 0.10 result is used as evidence for 0.11. Successful and accuracy-failing
modalities are distinguished below.

| Checkpoint | Successful tested modes | Executable modes with a quality failure |
| --- | --- | --- |
| google/gemma-4-E2B-it | Standalone text and audio; MTP text with its official assistant | Standalone image |
| google/gemma-4-E4B-it | Standalone image | Standalone text/audio; MTP text with its official assistant |
| google/gemma-4-12B-it | Standalone text and image; MTP, DSpark, EAGLE3 and both DFlash drafts in greedy text mode | Standalone audio |

The 12B companion checkpoints exercised are:

- `google/gemma-4-12B-it-assistant` (MTP).
- `deepseek-ai/dspark_gemma4_12b_block7` (DSpark).
- `deepseek-ai/eagle3_gemma4_12b_ttt7` (EAGLE3).
- `deepseek-ai/dflash_gemma4_12b_block7` and
  `z-lab/gemma4-12B-it-DFlash` (DFlash block7 and block16).

Successful greedy companion cases matched the independent target-model HF
reference; speculative-activity checks distinguish actual draft execution from
fallback. Quality failures remain runnable and remain failures against the
unchanged owning criteria. This table does not qualify every prompt, media
shape, context length, platform, combined image/audio request or generation
policy. Original-weight 26B/31B variants and their assistants are not established by
these smaller-model results.

## Sampling and capacity

MTP remains restricted to greedy requests. DSpark preserves mapped sampling
controls. EAGLE3 and DFlash V1 sampled requests complete through Edge 0.11's
vanilla fallback, but fail the unchanged positive-speculative-activity gate.
Model Connect forwards those executable requests unchanged with a warning;
it does not claim speculative sampling passed or silently force greedy output.

The persistent runtime reserves decode lookahead and checks engine capacity.
For media, Edge performs preprocessing before it can count expanded input
positions, so the adapter conservatively reserves the whole input profile plus
lookahead. The upstream runtime rejects media beyond that profile. Unsupported
options or insufficient capacity are errors, not ignored requests.

## Published 26B-A4B NVFP4 standalone route

```sh
trtmc gemma build /path/to/Gemma-4-26B-A4B-NVFP4 --precision fp16 \
  --task images_text_to_text --max-sequence-length 2048 -o gemma-nvfp4.bundle
```

The published checkpoint revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6` was built through Model Connect with
the original Edge 0.11 ONNX exporter and native SM120 text/visual builders.
Public Model Connect text and image commands completed. Direct Edge completed
the same-engine text and both original image fixtures, using 128 greedy tokens.
The unchanged upstream ROUGE-1/ROUGE-L gates (0.25/0.20) passed: text
0.3468/0.2197 and image 0.5825/0.3089.

**Quality remains unqualified:** Model Connect and direct Edge text and image outputs diverged
despite matching inspected greedy/thinking settings. The independent owning
HF FP32 NED comparison was not run because the installed reference loader does not
recognize this checkpoint's ModelOpt quantizer. Upstream ROUGE does not replace
that gate or resolve the divergence. These are executable requests, so the
adapter does not disable them for a quality failure. This evidence does not
qualify the 31B packed target, speculative companions, or other platforms.

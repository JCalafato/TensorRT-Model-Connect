# DiffusionGemma block diffusion

This family owns the documented Edge-LLM 0.11.0 checkpoint
`nvidia/diffusiongemma-26B-A4B-it-NVFP4`, pinned for validation to
`ec4ff3df205028f4e81c954c2227f9312b3ec2ea`.

## Inventory

| Prepared checkpoint | Inputs | Output | Execution |
| --- | --- | --- | --- |
| DiffusionGemma 26B-A4B NVFP4 | Text | Text | Block diffusion, batch 1, TP/CP 1 |
| Same checkpoint | One image and text | Text | Same encoder/denoiser plus vision encoder |
| Same checkpoint | Ordered images and text | Text | Separate image features and attention blocks |

This is not an autoregressive Gemma adapter. The family owns the shared-weight
encoder/denoiser, phase-specific layer scalars, self-conditioning, prefix cache,
expert quantization, sampler, vision graph, preprocessing, and final-answer
rendering. Audio, video, tool calls, speculative decoding, and trajectory outputs
are not advertised by this family. Unsupported request fields are rejected,
not silently discarded.

## Build and installation

```sh
python -m tensorrt_model_connect build /path/to/diffusiongemma-26B-A4B-it-NVFP4 \
  --family diffusion_gemma --task images_text_to_text \
  --precision fp16 --quantization nvfp4 --max-sequence-length 4096 \
  -o diffusiongemma.bundle
```

Build on the target GPU with a compatible native TensorRT installation. The
native path constructs the graph with TensorRT's Python API and executes it
through Model Connect's C++ runtime. It uses neither ONNX nor custom GPU kernels.
TensorRT owns GPU lowering; the family has no GPU-SKU allowlist. Engines still
require a compatible target/software stack and sufficient memory.

Prepared NVFP4 expert weights and calibration scales are consumed directly.
Builds do not launch a new calibration campaign or silently substitute a
different checkpoint. The recorded full-model execution used FP16 compute
and FP16 KV storage; a quantization metadata declaration alone is not evidence
of calibrated FP8 KV scales. The native builder also exposes FP32/BF16 compute,
but these are not full-checkpoint qualification claims.

Optional Edge tools may come from `tensorrt-edgellm==0.11.0` installed with pip or
from the existing source/CMake installation manifest. Installation origin does
not change request semantics. The official source pin is
`95515c2f87fba8982db5a519f9022277667b3cc9`.

**Known Edge command failure:** the pinned DiffusionGemma Python graph accesses
undefined `token_phase`. This was reproduced through its actual builder API on
a small graph, not presented as a completed full-checkpoint Edge build. The
family warns and uses the complete native path with the original request; it
does not advertise an Edge engine or repeatedly invoke the known-failing route.
All optional Edge discovery/selection is under this family's `edge_llm/` folder.

## Runtime contracts

The public C++ API exposes `TextContinuation` and `ImagesTextToText`. Vision
bundles retain both contracts; legacy text and single-image interfaces are also
available. Images are host HWC RGB uint8 or finite float32 pixels in `[0, 1]`.
The family preserves image order, projects each image independently, and assigns
distinct image-attention blocks. Image geometry and soft-token budget come from
the checkpoint, not a GPU-specific heuristic.

Typed options are `max_new_tokens`, `num_steps`, `seed`, `block_length`,
`confidence_threshold`, `use_chat_template`, `enable_thinking`, and
`system_prompt`. Media defaults to the provider template; text continuation
defaults to raw text. Chat image requests preserve ordered text/image parts,
an optional initial system/developer message, and user/assistant history.
Tool/reasoning input parts are explicitly unsupported. A raw image request
accepts one user message without a separate system instruction.

The checkpoint's canvas is 256 tokens. Capacity admission includes the **whole
last canvas**, not only the returned token budget. Every image and the expanded
request are checked before GPU execution. Native seeds are repeatable. Hidden
channel annotations are removed before decoding final answer text; original
generated token IDs remain available unchanged.

## Validation boundaries and remaining issues

The full prepared checkpoint has completed native build and actual text,
single-image, and two-image execution at capacity 4096, FP16 compute/source
NVFP4, TP1, 16 denoising steps, output budget 128, and seed 17. Six known-answer
requests were fixed before inference. The initial renderer exposed an empty
thought-channel label. After correcting it, a fresh full-model run through the
public C++ `Model::load`, `TextContinuation`, and `ImagesTextToText` paths passed
all six known-answer checks, including ordered two-image identification. This
run reused the successfully built engine; it is not a claim that the complete
pytest build-and-run profile was invoked a second time. An independent vLLM
0.24.0 run of the same prepared checkpoint also executed all eight requests and
passed the same six known answers, using the checkpoint's unchanged final-answer
rendering macro. That reference used BF16 compute/KV with Triton attention;
native used FP16 compute/KV. This validates the scoped answer quality, not
equal-precision logits, token identity, or open-ended prose equivalence.

Synthetic FP32 text/image/multi-image graphs, features, caches and generation
match an independent unchanged HF SDPA implementation. Exact preprocessing and
provider-template comparisons, seed repeatability, invalid-request recovery,
and text/image state-isolation checks also pass. These synthetic checks are not
catalog-wide or full-checkpoint accuracy evidence.

FP16 component comparisons still have recorded numerical discrepancies under
their unchanged tolerances. A standalone TensorRT comparison isolates
lowering-sensitive rounding at one explicit FP32 matrix-product/FP16-output
boundary; it does not establish a common cause for every component discrepancy.
The reviewed reference corrections are applied: HF eager attention receives
additive masks, and the expected same-image bidirectional mask is overlaid after
causal/window restrictions. Their FP32 checks pass. FP16 text and padded-vision
comparisons still fail their unchanged gates; independent alternative-backend
controls are not substituted for the owning golden.

The full-model executions above predate the latest text and vision
parameter-rounding corrections. No fresh full-checkpoint build or complete pytest qualification of
that corrected graph is claimed. No tolerance has been relaxed and no all-green
test suite is claimed. Accuracy-only discrepancies do not disable native execution.

The owning full-checkpoint profile requires all six fixed answers to match
after only case/whitespace normalization and terminal-punctuation removal:
arithmetic, world knowledge, image grounding, and ordered two-image animal
identification. It does not score open-ended prose as an exact-answer task or
replace the separate numerical component gates.

## Owning validation

```sh
cmake --build build --target trtmc_model_diffusion_gemma
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
TRTMC_RUNTIME_ROOT="$PWD/build" \
  PYTHONPATH=core/builder:apps/benchmark:. \
  python -m pytest families/diffusion_gemma/tests \
  --e2e-model diffusion_gemma -q
```

The E2E builds the prepared model and uses the public C++ model loader and Tasks.
`TRTMC_DIFFUSION_GEMMA_MODEL_DIR` can reuse an existing checkpoint, and
`TRTMC_DIFFUSION_GEMMA_ASSET_DIR` can reuse the pinned Edge example JPEGs.
Otherwise the owning test fetches the pinned checkpoint and public images from
the pinned Edge source revision. The test does not select a smaller substitute model on limited hardware.

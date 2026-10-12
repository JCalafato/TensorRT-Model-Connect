# Qwen3.6 / Qwen3.8 Edge-LLM execution

This family owns checkpoint admission, build options, command mapping, bundle
assets, runtime orchestration, and validation. Shared code contains no Edge
selection. The optional SDK uses official Edge-LLM **0.11.0**, revision
`95515c2f87fba8982db5a519f9022277667b3cc9`.

A compatible installed Python wheel is preferred. Discovery checks the actual
builder or ONNX exporter needed by the request, rather than accepting a version
string alone. Matching source-installed SDK Python is the fallback. Python
package discovery does not replace the native CMake SDK dependency. Builds are
native; cross compilation is not supported.

## Executed configurations

| Source checkpoint | Execution | Local build profile |
| --- | --- | --- |
| `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.8-27B` | Ordinary | Original weights, FP16, TP2, batch 1, Linux x86_64 SM80 |
| `RadixArk/Qwen3.8-27B-NVFP4` | Ordinary | Original mixed FP8/NVFP4 weights, FP16 execution, TP1, batch 1, Linux x86_64 SM120 |
| Same mixed checkpoint + `RadixArk/Qwen3.8-27B-DSpark` | DSpark | Unquantized block-7 draft, text, FP16, TP1, batch 1, SM120 |
| Same mixed checkpoint + `z-lab/Qwen3.8-27B-DFlash2` | DFlash2 | Unquantized block-8 draft, text, FP16, TP1, batch 1, SM120 |

Ordinary media builds expose existing image, ordered-image, video, and mixed
media Task interfaces. Availability of a binding does not qualify every media
workload. Current executed evidence and remaining gaps are listed below.

### Additional upstream execution mappings

The adapter also maps the following Edge 0.11 configurations. These have owning
Python and native contract-test coverage, but **no local model-engine execution
or accuracy result**:

- Qwen3.6 uniform NVFP4 with embedded MTP.
- Qwen3.6 INT4-AWQ with embedded MTP.
- Qwen3.8 INT4-AWQ with the documented DFlash2 companion.

These mappings require compatible ModelOpt checkpoint metadata: uniform NVFP4
with group size 16, or INT4-AWQ with group size 128. They do not accept every
checkpoint whose name contains NVFP4 or INT4. The adapter forwards prepared
weights to Edge; it does not perform calibration or create quantized checkpoints.
The locally executed table above is evidence scope, not a checkpoint-ID allowlist.

## Build and run

Provision the [native SDK](../../../cmake/edge_llm/README.md), configure its
installation in `CMAKE_PREFIX_PATH`, and compile with `TRTMC_ENABLE_EDGELLM=ON`.
The family options use the existing `cli.json` protocol, not a shared parser
extension. Help works offline:

```sh
trtmc qwen3_8 build --help

CUDA_VISIBLE_DEVICES=0,1 trtmc qwen3_8 build /models/Qwen3.6-27B \
  --precision fp16 --tensor-parallel-size 2 --max-sequence-length 4096 \
  -o qwen36.bundle

trtmc qwen3_8 build /models/Qwen3.8-27B-NVFP4 \
  --task vision_language_generation --precision fp16 --quantization nvfp4 \
  --max-sequence-length 4096 -o qwen38-media.bundle

trtmc run qwen38-media.bundle --image /path/to/image.png \
  --prompt "Please describe the image." --max-new-tokens 128 \
  --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false

trtmc qwen3_8 build /models/Qwen3.8-27B-NVFP4 --precision fp16 \
  --execution-variant dspark --companion draft=/models/Qwen3.8-27B-DSpark \
  --max-sequence-length 512 -o qwen38-dspark.bundle

trtmc qwen3_8 build /models/Qwen3.8-27B-NVFP4 --precision fp16 \
  --execution-variant dflash --companion draft=/models/Qwen3.8-27B-DFlash2 \
  --max-sequence-length 512 -o qwen38-dflash2.bundle
```

For the additional mapped profiles, use a compatible prepared checkpoint:

```sh
trtmc qwen3_8 build /models/Qwen3.6-27B-NVFP4 --precision fp16 \
  --quantization nvfp4 --execution-variant mtp \
  --max-sequence-length 4096 -o qwen36-mtp.bundle

trtmc qwen3_8 build /models/Qwen3.8-27B-INT4-AWQ --precision fp16 \
  --quantization int4_awq --execution-variant dflash \
  --companion draft=/models/Qwen3.8-27B-DFlash2 \
  --max-sequence-length 512 -o qwen38-int4-dflash2.bundle
```

MTP uses the checkpoint's embedded draft, with three draft steps and four
verification positions; it takes no `--companion`. Its INT4-AWQ variant uses
`--quantization int4_awq` and a matching checkpoint. These examples describe the
mapped interface, not additional locally executed model tests.

DFlash2 uses Edge's public `dflash` mode. Paired builds invoke the official
ONNX exporter and native engine builder for both engines. They include required
embedding, selector/head, tokenizer, and prompt-template assets. Runtime
verification uses size 8, drafting top-K 1/step 1, and DSpark scheduling off;
the draft architecture selects the appropriate block semantics. DFlash2 DDTree
is not supported. The mapped DSpark context range is 9–1024 tokens; DFlash2 is
9–4096, also bounded by source metadata.

Ordinary builds map the official Python builder and native C++ runtime. TP2
uses one process with Edge-managed rank threads and requires two matching GPUs
and compatible NCCL. Mixed-NVFP4 uses `--dense auto`, preserving the checkpoint's
actual per-layer and KV-scale metadata instead of requantizing or forcing a
precision. Image profiles allow 128–1280 tokens per image and size the total
visual capacity to the LLM input budget. Edge owns pixel preprocessing.

Self-contained bundles include only original checkpoint shards named by emitted
runtime metadata, plus their unchanged index. Use scratch output and `TMPDIR`:
publication and extraction temporarily coexist with large bundles. Keep source
weights immutable through build completion.

Build preparation failures warn and attempt the unchanged native request once.
Native rejection remains explicit for unmapped TP2 or speculative requests; a
failed pair never becomes ordinary base-only decoding. Runtime errors propagate
without fallback. Unsupported controls are rejected before execution. Accuracy
failures are evidence, **not** a dispatch denylist.

## Local evidence and limits

All comparisons retain the owning NED **0.15** gate. Exact agreement with direct
Edge is transport parity, not an independent quality result.

| Exact workload | Build/inference | Independent quality |
| --- | --- | --- |
| Qwen3.6 original FP16 TP2, raw text, 16 greedy tokens | Model Connect and Edge passed, exact agreement | NED 0.0, but the unchanged no-thinking-markup predicate failed; the FP32 reference also emits the markup |
| Qwen3.6 original FP16 TP2, capital chat, 10 tokens | Model Connect passed | Fresh FP32 comparison passed, NED 0.0 |
| Qwen3.6 original FP16 TP2, single image, 128 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.23063 |
| Qwen3.6 original FP16 TP2, ordered two images, 128 tokens | Both passed, exact agreement | NED 0.06217 passes, but the required giant-panda content check fails |
| Qwen3.6 original FP16 TP2, two-frame video transition, 64 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.63754 |
| Qwen3.8 original FP16 TP2, same raw text | Both passed, exact agreement | Fresh FP32 comparison passed, NED 0.0 |
| Qwen3.8 original FP16 TP2, capital chat, 10 tokens | Model Connect passed | Fresh FP32 comparison passed, NED 0.0 |
| Qwen3.8 original FP16 TP2, single image, 128 tokens | Both passed, exact agreement | Fresh FP32 comparison and woman/dog/beach content checks passed, NED 0.0 |
| Qwen3.8 original FP16 TP2, ordered two images, 128 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.19279 |
| Qwen3.8 original FP16 TP2, two-frame video transition, 64 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.62058 |
| Qwen3.8 mixed NVFP4, ordinary/DSpark/DFlash2, raw text, 16 tokens | All three passed, exact direct agreement | All failed NED 0.21875 against the same checkpoint decoded to FP32 |
| Qwen3.8 mixed NVFP4 ordinary, single/ordered images | Both passed, exact direct agreement | Failed matched packed-weight FP32 comparisons: NED 0.27907 / 0.77639; ordered-image giant-panda content check also failed |
| Qwen3.8 mixed NVFP4 ordinary, two-frame video transition | Both passed, exact direct agreement | Failed matched packed-weight FP32 comparison, NED 0.70667 |

The mixed-NVFP4 ordinary, DSpark, and DFlash2 capital-chat requests passed fresh same-checkpoint
FP32 comparisons (NED 0.0, 10-token budget). This does not override their raw-text
failure or establish the registered original-checkpoint E2E/benchmark result.
The FP32 mixed-weight oracle explicitly decodes original ModelOpt weights; it
does not emulate compiled activation or KV quantization. A difference is not,
by itself, evidence of a TensorRT compiler bug.

Empty-video rejection and a successful image request afterward were tested.
Video evidence currently covers a synthetic two-frame animal transition, not
general motion understanding. No quality criterion was relaxed.

Tested source revisions:

- Qwen3.6 original: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Qwen3.8 original: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Mixed NVFP4: `319f741cce68d7914884900c138a1fbb70a42f30`.
- DSpark: `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
- DFlash2: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

Original-weight DFlash2 is not admitted: FP16 text and draft source weights alone
total about 57.6 GB, above the 48 GiB capacity of the largest available individual
validation device, before runtime overhead. The pinned ONNX builder rejects
speculative TP greater than one. This is a conservative capacity-admission
estimate, not an executed GPU OOM or a measured TensorRT engine allocation.

The additional INT4 DFlash2 and uniform NVFP4/INT4-AWQ MTP mappings have not had
local engine build, inference, or accuracy validation. They delegate to the
documented upstream flows; unit and native contract tests do not replace those
missing model results. Reproducing every upstream calibration recipe is not a
prerequisite for using an already-compatible checkpoint or publishing this
integration. The original media builds retain unchanged FP32/content gates.
DSpark remains mixed-NVFP4-only; uniform
INT4-AWQ is admitted only for DFlash2 and MTP, never ordinary execution.
These results do not claim catalog-wide or registered CI coverage.

The existing original-checkpoint premerge manifests use FP16 TP1, not the TP2
profiles above. Their full E2E runs and registered MMLU/performance benchmarks
remain unexecuted; the capital-chat comparison is not a replacement for those
suites. Original FP16 single-device validation needs a larger device than the
available 48 GiB maximum. This is a resource-admission limit, not an observed
Edge command failure.

The official performance tables also contain batch-8 MTP rows. This adapter
currently maps only batch 1; it does not implement or claim the batched profile.
Other published platform rows are not qualified by the recorded SM80/SM120
results.

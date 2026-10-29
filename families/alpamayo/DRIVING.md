# Alpamayo driving record

The family adapters use the existing `IStructuredDecision` task. All driving
semantics, JSON validation, image preparation, reasoning, action integration and
physical decoding belong to this family. No shared task or robot-action change
is required. This interface description is not a model qualification claim.

## Input

Set `StructuredDecisionRequest.document` to a JSON object with:

- `schema`: `alpamayo.driving.v1`.
- `coordinate_frame`: `current_ego`; history positions are metres and rotations
  are row-major 3-by-3 rotation matrices in the current ego frame.
- `observations`: nonempty array of objects containing a unique `id`, `history`
  and `cameras`.
- `num_samples`: 1 (default) or 6 per observation, bounded by the bundle's batch
  capacity. Rows are observation-major, then sample-major.
- Optional `sampling`: `seed` (nonnegative signed 64-bit integer), `temperature`,
  `top_p`, `top_k`, `max_new_tokens`, and `denoising_steps` (10). Defaults are the
  driving provider's 0.6 temperature, 0.98 top-p, disabled top-k and ten Euler
  steps, not the backbone's chat-generation defaults.
- Optional `initial_noise`: `[total_samples,64,2]` finite FP32 values for matched
  noise comparisons. Otherwise native Gaussian sampling is repeatable within
  the same runtime, without promising PyTorch or cross-standard-library RNG
  identity.

Each history contains `xyz[16][3]`, `rotations[16][3][3]` and
`timestamps_us[16]`. Timestamps are nonnegative, strictly increasing and sampled
at the trained 0.1-second interval. Each camera object contains `camera_index`,
`video_index` and four `timestamps_us`. Camera indices are ordered `[0,1,2,6]`;
frames are chronological within each camera. Camera times cannot follow the
current history pose. Times validate ordering; they are not additional learned
model features.

Provide exactly four decoded camera videos per observation through
`StructuredDecisionRequest.videos`. Each is four contiguous THWC RGB frames,
with integer-valued float pixels in `[0,255]`. Camera objects refer to distinct
video entries. Do not supply unordered still images or `max_state_tokens`.
Invalid records are rejected before engine execution; there is no implicit
truncation, camera reordering, image resampling to fit capacity, or pose omission.

## Output

`StructuredDecisionResult.document` contains schema
`alpamayo.driving.result.v1`, coordinate frame `current_ego`, and `samples`.
Every sample preserves its observation `id` and `sample_index` and includes:

- Generated token IDs and separate `cot`, `meta_action`, and `answer` text.
- `normalized_acceleration_curvature[64][2]`, explicitly not physical poses.
- Physical `trajectory.xyz[64][3]`, `trajectory.rotations[64][3][3]`, future
  `timestamps_us[64]`, and `initial_velocity_mps`.

Empty generic decision scores are intentional: driving trajectories are not
classification probabilities. The structured JSON document is the complete
family-owned result, not a text-only substitute.

## Qualification boundary

The native bundle contains independent `vision.plan`, `text.plan`, `action.plan`,
`runtime.json`, and the trained extended tokenizer. An engine build failure
aborts the bundle. The existing native component accuracy criteria are unchanged.
Automatic discovery selects the complete structured-decision task. With compatible
installed Edge 0.11 tools, `auto` builds all visual, reasoning and action components
using the official direct builder. Missing/incompatible tools or a builder failure
warn and fall back to the native TensorRT graph; a partially written bundle is
never published. Runtime errors are surfaced, not silently retried.

Edge output masking uses the official reduced-vocabulary builder option, preserving
all input embeddings while excluding the provider's discrete trajectory output
tokens. A family-owned template preserves ego history and the trained reasoning
continuation. Physical trajectory decoding remains family-owned.

Successful commands do **not** imply validated accuracy. The observed Edge driving
reasoning remains inaccurate; the route stays runnable for investigation. Existing
native component and public-example quality criteria are unchanged.

## Build entry points

Both existing command paths reach the same family-owned builder:

```bash
trtmc build nvidia/Alpamayo-R1-10B -o driving.bundle --max-batch-size 1
trtmc alpamayo build nvidia/Alpamayo-R1-10B -o driving.bundle --max-batch-size 6
trtmc alpamayo build nvidia/Alpamayo-R1-10B -o native.bundle --execution-variant native
```

The family-only `--execution-variant` option accepts `auto` (default) or `native`.
Use `native` for explicit `initial_noise` or seeds larger than signed 32-bit:
Edge's public action API exposes only its int32 noise seed, not input noise tensors.
Other request validation and the complete output schema are shared within this family.
Compatible wheel Python tools may be used with the CMake-installed Edge C++ SDK;
source-built installations use the same contract. Standard `CMAKE_PREFIX_PATH`
discovery is used, and the adapter does not install packages during a model build.

The released checkpoint omits its backbone tokenizer and processor metadata.
A complete local `base_vlm/` is preferred. Otherwise this family resolves only
four metadata/tokenizer files from the pinned Qwen3-VL backbone revision; no
backbone weights are downloaded and no standalone Qwen adapter is registered.
Standard Hugging Face cache and offline settings apply. Incomplete explicit
local assets are rejected instead of silently mixing revisions.

## Public-example E2E

`alpamayo-r1-fp16-driving-b6` builds the complete native bundle, submits four
four-frame videos and ego history through the existing public worker, and
compares six 6.4-second physical trajectories with independently recorded
PhysicalAI-AV ground truth. It checks the documented minADE6 limit of 0.90 m
on one pinned public example, not the broader qualification set.
Component numerical tests remain separate and unchanged.

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/alpamayo/tests/test_e2e.py \
  --e2e-testcase alpamayo-r1-fp16-driving-b6 -vv
```

Set `TRTMC_RUNTIME_ROOT` to the built runtime directory. The test uses its
`trtmc_benchmark_worker`, or `TRTMC_BENCHMARK_WORKER` when separately installed.
`TRTMC_ALPAMAYO_MODEL_DIR` reuses a local checkpoint; `TRTMC_ALPAMAYO_SAMPLE_DIR`
reuses the independently prepared public example after pinned provenance, raw-frame structure
and state-array layouts are verified. Otherwise the pinned public provider data loader
streams only the selected clip, refusing unbounded archive downloads. Access to
the public dataset still requires its normal Hugging Face authorization.

# Alpamayo driving reasoning and trajectories

This family owns `nvidia/Alpamayo-R1-10B` at revision
`dd4a24cacefc9a6477a6dfc7354de2443401409d`, the FP16 configuration documented
by Edge-LLM 0.11.0. Documented sample capacities are one and six, on one device.
The complete path consumes four cameras with four frames each and sixteen
historical ego poses. It returns reasoning and 64 future physical poses at
100 ms intervals. It is not a text-only substitute or a path requiring callers
to supply precomputed visual features or attention caches.

## Build and execution selection

The existing generic build command discovers the family from checkpoint metadata:

```sh
python -m tensorrt_model_connect build /path/to/Alpamayo-R1-10B \
  --max-batch-size 6 --max-sequence-length 4096 -o alpamayo.bundle
```

The family command also exposes explicit native selection:

```sh
python -m tensorrt_model_connect alpamayo build /path/to/Alpamayo-R1-10B \
  --execution-variant native --max-batch-size 6 \
  --max-sequence-length 4096 -o alpamayo-native.bundle
```

`auto` uses compatible installed Edge-LLM 0.11.0 Python tools and native SDK
when available. Wheel and source installations use the same discovery contract;
construction does not install dependencies. All Edge adapters remain under this
family's `edge_llm/` directories. Edge preparation errors preserve diagnostics,
warn, and retry native construction once with the unchanged request. Bundle
publication and inference errors are surfaced, not silently retried on another
backend. Accuracy-only failures do not disable a runnable Edge path.

Native construction defines vision, language and action graphs with the TensorRT
Python API. The Model Connect C++ runtime owns preprocessing, autoregressive
reasoning, prefix-cache handoff, flow integration and physical trajectory output.
It requires neither ONNX nor custom GPU kernels. There is no family GPU allowlist;
optional SDK compatibility and TensorRT capability determine the available path.

## Public runtime contract

The bundle implements `IStructuredDecision` through the existing Model Connect
Task interface. The request document uses schema `alpamayo.driving.v1` and
coordinate frame `current_ego`, with camera/video indices, frame timestamps,
history XYZ/rotations/timestamps, and a sample count within bundle capacity.
Raw frames are provided through the Task's video inputs.

The response schema is `alpamayo.driving.result.v1`. Each sample includes reasoning
tokens and text, normalized acceleration/curvature, and physical XYZ, rotation
matrices and timestamps. Native requests accept an explicit noise tensor or an
integer seed. The Edge adapter checks its runtime's narrower signed 32-bit seed
contract and does not pretend to support explicit native noise injection.

## Reproducible validation and limits

The family-owned manifest pins the public checkpoint, provider loader and one
public driving clip. Run its complete native build and public C++ Task test with:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/alpamayo/tests/test_e2e.py \
  --e2e-testcase alpamayo-r1-fp16-driving-b6 -vv
```

Set `TRTMC_RUNTIME_ROOT` to a build containing the family DSO and
`trtmc_benchmark_worker`. Optional `TRTMC_ALPAMAYO_MODEL_DIR` and
`TRTMC_ALPAMAYO_SAMPLE_DIR` select already prepared local assets. The test does
not select Edge merely because an SDK is installed. Its independent trajectory
criterion is minADE6 at 6.4 seconds at most 0.90 m; component numerical gates
remain separate and unchanged.

Actual native B1 and B6 construction and raw-media Task execution succeeded.
The B6 public-clip evaluation measured 0.7304 m. This is a single-clip result,
not reproduction of the broader benchmark or safety qualification.
The registered pytest wrapper has not yet been rerun end to end; recorded public
build/Task commands and the same owning evaluator produced this scoped result.

Current blockers are intentionally retained:

- FP16 native text and vision component assertions fail at the unchanged
  `rtol=0.002`, `atol=0.0002` gates. The text discrepancy is associated with
  TensorRT fused-attention lowering; an observed-score diagnostic changes the
  lowering and agrees exactly with the reference, but is not a production fix.
  The vision discrepancy has a separate first differing projection. These are
  accuracy failures, not command or memory failures.
- Direct Edge and Model Connect Edge B1/B6 commands execute on a tested x86
  configuration, but reasoning quality is incorrect. Model Connect Edge B6
  measured 3.0877 m on the same clip, failing the 0.90 m criterion. This runnable
  route is not presented as accuracy-qualified.
- A source-SDK compiler failure on a tested ARM configuration is recorded
  separately; it is not a model-wide or native GPU restriction.

No passing result here establishes all-platform, all-precision, or autonomous
vehicle deployment suitability. These unresolved gates prevent a clean merge
qualification claim.

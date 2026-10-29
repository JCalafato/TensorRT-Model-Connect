# DiffusionGemma and Edge-LLM 0.11.0

The official 0.11.0 catalog lists `nvidia/diffusiongemma-26B-A4B-it-NVFP4`
for text/image-to-text block diffusion. The pinned direct builder at revision
`95515c2f87fba8982db5a519f9022277667b3cc9` currently references the undefined
variable `token_phase` in `DiffusionGemma` text graph construction. The error
was reproduced by calling the unmodified builder API with synthetic weights;
this is not evidence of a full-checkpoint build or a quality assessment.

This family therefore does not publish an Edge engine for that release. It
keeps the complete native TensorRT request available, including its image
encoder, text encoder, diffusion denoiser, and prepared quantized weights.
The native builder uses TensorRT Python graphs and the family-owned C++ runtime,
not ONNX or custom GPU kernels. Accuracy-only results do not select or reject
a backend. No GPU-specific restriction is imposed on the native graph.

Optional Edge installations are detected through the standard
`tensorrt-edgellm` wheel metadata or a source-built CMake installation's
`share/trtmc/edge-llm.json` in `CMAKE_PREFIX_PATH`. Discovery does not install
dependencies or import an incompatible CUDA runtime. Both installation origins
receive the same explicit warning and native route for the pinned upstream
failure. A future upstream fix requires separate validation before enabling
Edge offload; installing another release alone does not enable it.

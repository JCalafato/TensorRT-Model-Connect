# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT math with explicit provider precision boundaries."""

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, weights, precision):
        if precision not in ("fp32", "fp16"):
            raise ValueError("Alpamayo native graphs require fp32 or fp16 precision")
        self.net, self.weights = network, weights
        self.dtype = trt.float32 if precision == "fp32" else trt.float16
        self.storage = np.float32 if precision == "fp32" else np.float16
        self.buffers = []
        self.constants = {}

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.net.add_cast(value, dtype).get_output(0)

    def const(self, array):
        array = np.ascontiguousarray(array)
        self.buffers.append(array)
        return self.net.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def scalar(self, value, rank):
        return self.const(np.full((1,) * rank, value, np.float32))

    def weight(self, name, shape):
        if tuple(self.weights.info(name)[1]) != tuple(shape):
            raise ValueError(f"Unexpected Alpamayo tensor shape for {name}: expected {shape}")
        if name not in self.constants:
            self.constants[name] = self.const(self.weights.array(name, self.storage))
        return self.constants[name]

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, left, right, operation):
        return self.net.add_elementwise(left, right, operation).get_output(0)

    def concat(self, values, axis):
        layer = self.net.add_concatenation(values)
        layer.axis = axis
        return layer.get_output(0)

    def linear(self, value, name, out_features, *, bias=True):
        rank, width = len(value.shape), value.shape[-1]
        weight = self.weight(name + ".weight", (out_features, width))
        weight = self.reshape(weight, (1,) * (rank - 2) + (out_features, width))
        # Autocast rounds the operands, accumulates the affine operation in
        # FP32, then rounds its output. Bias belongs before that final cast.
        value = self.cast(self.cast(value, self.dtype), trt.float32)
        product = self.net.add_matrix_multiply(
            value,
            trt.MatrixOperation.NONE,
            self.cast(weight, trt.float32),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        if not bias:
            return self.cast(product, self.dtype)
        bias = self.weight(name + ".bias", (out_features,))
        bias = self.reshape(bias, (1,) * (rank - 1) + (out_features,))
        result = self.op(product, self.cast(bias, trt.float32), trt.ElementWiseOperation.SUM)
        return self.cast(result, self.dtype)

    def silu(self, value):
        fp32 = self.cast(value, trt.float32)
        sigmoid = self.net.add_activation(fp32, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.op(fp32, sigmoid, trt.ElementWiseOperation.PROD), value.dtype)

    def rms_norm(self, value, name, epsilon):
        rank, width = len(value.shape), value.shape[-1]
        fp32 = self.cast(value, trt.float32)
        square = self.op(fp32, fp32, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(
            square, trt.ReduceOperation.AVG, 1 << (rank - 1), True
        ).get_output(0)
        variance = self.op(mean, self.scalar(epsilon, rank), trt.ElementWiseOperation.SUM)
        inverse = self.op(variance, self.scalar(-0.5, rank), trt.ElementWiseOperation.POW)
        normalized = self.op(fp32, inverse, trt.ElementWiseOperation.PROD)
        # The provider's action RMSNorm casts BEFORE multiplying its learned scale.
        normalized = self.cast(normalized, value.dtype)
        weight = self.reshape(self.weight(name + ".weight", (width,)), (1,) * (rank - 1) + (width,))
        return self.op(normalized, self.cast(weight, value.dtype), trt.ElementWiseOperation.PROD)

    def layer_norm(self, value, name, epsilon):
        rank, width = len(value.shape), value.shape[-1]
        value = self.cast(value, trt.float32)
        shape = (1,) * (rank - 1) + (width,)
        weight = self.reshape(self.weight(name + ".weight", (width,)), shape)
        bias = self.reshape(self.weight(name + ".bias", (width,)), shape)
        # CUDA autocast LayerNorm returns FP32, including its affine parameters.
        layer = self.net.add_normalization(
            value, self.cast(weight, trt.float32), self.cast(bias, trt.float32), 1 << (rank - 1)
        )
        layer.epsilon = epsilon
        return layer.get_output(0)

#!/usr/bin/env python3
"""Creates tiny stand-in models with the same input/output as the real one.

They make the end-to-end tests fast and deterministic; a separate test runs
the real model. Output: tests/fixtures/mock-model/{default,gpu-fail}/.

- default:  alpha = sigmoid(4 * mean over RGB of the normalised input)
            (bright areas become opaque, dark areas transparent)
- gpu-fail: the same, plus a Split into 12 outputs. WebGPU implementations
            that allow fewer storage buffers per shader reject it at run time,
            which exercises the automatic WebGPU → WebAssembly fallback.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mock-model"
SIZE = 1024


def build(gpu_fail: bool) -> onnx.ModelProto:
    nodes = [
        helper.make_node("ReduceMean", ["input_image"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["mean", "gain"], ["scaled"]),
    ]
    inits = [
        numpy_helper.from_array(np.array(4.0, np.float32), "gain"),
    ]
    last = "scaled"
    if gpu_fail:
        # A Split into 12 outputs needs 13 storage buffers in one shader,
        # more than common WebGPU implementations allow.
        n = 12
        widths = [SIZE // n] * (n - 1) + [SIZE - (SIZE // n) * (n - 1)]
        inits.append(numpy_helper.from_array(np.array(widths, np.int64), "widths"))
        outs = [f"p{i}" for i in range(n)]
        nodes.append(helper.make_node("Split", ["scaled", "widths"], outs, axis=3))
        parts = []
        for i in range(n):
            # Slightly different factors keep ONNX Runtime from merging or removing the branches.
            inits.append(numpy_helper.from_array(np.array(1.0 + i * 1e-6, np.float32), f"f{i}"))
            nodes.append(helper.make_node("Mul", [outs[i], f"f{i}"], [f"m{i}"]))
            parts.append(f"m{i}")
        nodes.append(helper.make_node("Concat", parts, ["joined"], axis=3))
        last = "joined"
    nodes.append(helper.make_node("Sigmoid", [last], ["alpha"]))
    graph = helper.make_graph(
        nodes,
        "mock",
        [helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [1, 1, SIZE, SIZE])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="mock")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    for name, gpu_fail in (("default", False), ("gpu-fail", True)):
        data = build(gpu_fail).SerializeToString()
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.onnx.part00").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        manifest = {
            "id": f"mock-{name}",
            "version": "1",
            "name": f"Mock model ({name})",
            "task": "testing only",
            "input": {"name": "input_image", "width": SIZE, "height": SIZE,
                      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "output": {"name": "alpha", "width": SIZE, "height": SIZE},
            "weights": "fp32",
            "size": len(data),
            "sha256": digest,
            "parts": [{"file": "model.onnx.part00", "size": len(data), "sha256": digest}],
            "source": {"url": "tools/testing/make_mock_models.py", "sha256": digest, "license": "MIT", "homepage": ""},
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {len(data)} bytes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the browser model from the official BiRefNet_lite ONNX release.

Source: https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1
        BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx (MIT License, (c) 2024 ZhengPeng)

The official export is numerically fine but not usable in a browser as-is:
its deformable convolutions were exported as a GatherND/ScatterND
decomposition that materialises tensors of up to 784 MB each and needs
~12 GB of RAM for a single 1024x1024 inference.

This script performs documented, verifiable transformations:

1. Deformable convolutions: every `.../atrous_conv` subgraph is replaced by a
   mathematically equivalent, memory-lean formulation (bilinear sampling via
   Gather on a zero-padded input, one corner and a small group of kernel taps
   at a time, accumulated with MatMul). The offset/modulator convolutions are
   kept as-is.
2. The decoder's image-to-patches step (a Concat of up to 1024 inputs) becomes
   Reshape → Transpose → Reshape, which WebGPU can run.
3. Memory rewrites that never materialise the decoder's widest tensors (up to
   480 MiB at 1024 × 1024): a pointwise convolution over a concatenation
   becomes a sum of pointwise convolutions, a pointwise convolution after a
   bilinear resize moves before it, a pointwise convolution after another
   convolution is folded into it, and a huge convolution → convolution
   intermediate is computed in channel chunks. All are exact linear algebra.
4. The raw logits output gets a Sigmoid so the model directly returns an
   alpha matte in [0, 1] named "alpha".
5. Weight storage: large float32 weights are stored as float16 (default) and
   converted back to float32 by a Cast node, so all computation stays float32
   while the download is roughly halved.

Run `python tools/model/convert_birefnet.py --help` for options. The output is
split into chunks (default 24 MiB) plus a manifest with SHA-256 hashes that
the web app uses to download, verify and cache the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

SOURCE = {
    "name": "BiRefNet_lite (general, Swin-T backbone, epoch 232)",
    "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/"
    "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
    "sha256": "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
    "size": 224005088,
    "license": "MIT",
    "copyright": "Copyright (c) 2024 ZhengPeng",
    "homepage": "https://github.com/ZhengPeng7/BiRefNet",
}

# SHA-256 of the complete converted model (default options) as validated
# against the original export (see docs/MODEL.md). The conversion is
# deterministic; if this changes, re-run the numerical comparison first.
VALIDATED_OUTPUT_SHA256 = {
    "fp16": "d816fd9efeac9a0c5ec15e579d83a21dba901327465516a4d83ae084dd5325bf",
}

INPUT_NAME = "input_image"
OUTPUT_NAME = "alpha"
INPUT_SIZE = 1024
# Upper bound for the gathered samples of one group of kernel taps (one corner).
GROUP_BYTES = 16 * 1024 * 1024
# Pointwise convolutions over concatenations at least this large are split.
SPLIT_MIN_BYTES = 32 * 1024 * 1024
# Conv → Conv intermediates larger than this are computed in channel chunks.
CHUNK_MAX_BYTES = 64 * 1024 * 1024
# Only weights with at least this many elements are stored in reduced precision.
MIN_COMPRESS_ELEMENTS = 4096

MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2024 ZhengPeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_source(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / SOURCE["url"].rsplit("/", 1)[-1]
    if not target.exists() or target.stat().st_size != SOURCE["size"]:
        log(f"Downloading {SOURCE['url']}")
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(SOURCE["url"]) as resp, tmp.open("wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
        tmp.replace(target)
    digest = sha256_file(target)
    if digest != SOURCE["sha256"]:
        raise SystemExit(
            f"SHA-256 mismatch for {target}: expected {SOURCE['sha256']}, got {digest}"
        )
    log(f"Verified source model {target.name} ({digest})")
    return target


class GraphEditor:
    """Small helper around an ONNX graph for adding nodes and constants."""

    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.graph = model.graph
        self.new_nodes: list[onnx.NodeProto] = []
        self.new_inits: list[onnx.TensorProto] = []
        self._uid = 0

    def name(self, prefix: str) -> str:
        self._uid += 1
        return f"{prefix}_{self._uid}"

    def const(self, prefix: str, value: np.ndarray) -> str:
        n = self.name(prefix)
        self.new_inits.append(numpy_helper.from_array(value, n))
        return n

    def node(self, op: str, inputs: list[str], prefix: str, output: str | None = None, **attrs) -> str:
        out = output or self.name(prefix + "/" + op)
        self.new_nodes.append(helper.make_node(op, inputs, [out], name=self.name(prefix + "/" + op), **attrs))
        return out


def infer_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = shape_inference.infer_shapes(model, data_prop=True)
    shapes: dict[str, list[int]] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            shapes[vi.name] = [d.dim_value for d in dims]
    return shapes


def replace_deformable_convs(model: onnx.ModelProto) -> int:
    graph = model.graph
    shapes = infer_shapes(model)
    inits = {t.name: t for t in graph.initializer}

    scopes: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for n in graph.node:
        m = re.match(r"^(.*/atrous_conv)/", n.name)
        if m:
            scopes[m.group(1)].append(n)
    if not scopes:
        raise SystemExit("No deformable convolution scopes found – unexpected model layout")

    ed = GraphEditor(model)
    remove: set[int] = set()  # ids of NodeProto objects

    for scope, nodes in scopes.items():
        by_name = {n.name: n for n in nodes}
        offset_conv = by_name[f"{scope}/offset_conv/Conv"]
        modulator_conv = by_name[f"{scope}/modulator_conv/Conv"]
        sigmoid = by_name[f"{scope}/Sigmoid"]
        mask_mul = by_name[f"{scope}/Mul"]
        regular = by_name[f"{scope}/Conv"]
        final_add = by_name[f"{scope}/Add_3"]
        assert sigmoid.input[0] == modulator_conv.output[0]
        assert mask_mul.input[0] == sigmoid.output[0]

        x = offset_conv.input[0]
        offset = offset_conv.output[0]
        mask = mask_mul.output[0]
        out_name = final_add.output[0]

        weight = numpy_helper.to_array(inits[regular.input[1]]).astype(np.float32)
        cout, cin, kh, kw = weight.shape
        if kh != kw:
            raise SystemExit(f"{scope}: non-square kernel not supported")
        k_count = kh * kw
        pad = kh // 2

        # Bias of the regular conv is exported as a constant added afterwards.
        bias = None
        for inp in final_add.input:
            if inp == regular.output[0]:
                continue
            producer = next((n for n in nodes if inp in n.output), None)
            if producer is not None and producer.op_type == "Constant":
                bias = numpy_helper.to_array(producer.attribute[0].t).astype(np.float32)
            elif inp in inits:
                bias = numpy_helper.to_array(inits[inp]).astype(np.float32)
        if bias is not None and not np.any(bias):
            bias = None

        _, c_x, h, w = shapes[x]
        assert c_x == cin, (scope, c_x, cin)
        hw = h * w
        hp, wp = h + 2, w + 2
        p = scope + "/dcn"

        # Zero-pad by one pixel: out-of-range samples then read zeros, exactly
        # like the original export (and torchvision.ops.deform_conv2d).
        pads = ed.const(p + "/pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64))
        xp = ed.node("Pad", [x, pads], p, mode="constant")
        xf = ed.node("Reshape", [xp, ed.const(p + "/shape", np.array([cin, hp * wp], np.int64))], p)

        off = ed.node("Reshape", [offset, ed.const(p + "/shape", np.array([k_count, 2, hw], np.int64))], p)
        dy = ed.node("Gather", [off, ed.const(p + "/i", np.array(0, np.int64))], p, axis=1)
        dx = ed.node("Gather", [off, ed.const(p + "/i", np.array(1, np.int64))], p, axis=1)

        # Pixel grid, generated at load time instead of being stored in the file.
        zero = ed.const(p + "/zero", np.array(0.0, np.float32))
        one = ed.const(p + "/one", np.array(1.0, np.float32))
        ry = ed.node("Range", [zero, ed.const(p + "/h", np.array(float(h), np.float32)), one], p)
        rx = ed.node("Range", [zero, ed.const(p + "/w", np.array(float(w), np.float32)), one], p)
        hw_shape = ed.const(p + "/hw", np.array([h, w], np.int64))
        gy = ed.node("Expand", [ed.node("Reshape", [ry, ed.const(p + "/s", np.array([h, 1], np.int64))], p), hw_shape], p)
        gx = ed.node("Expand", [ed.node("Reshape", [rx, ed.const(p + "/s", np.array([1, w], np.int64))], p), hw_shape], p)
        flat = ed.const(p + "/s", np.array([1, hw], np.int64))
        gy = ed.node("Reshape", [gy, flat], p)
        gx = ed.node("Reshape", [gx, flat], p)

        taps = np.arange(k_count)
        ky = (taps // kw - pad + 1).astype(np.float32).reshape(k_count, 1)  # +1: padded coordinates
        kx = (taps % kw - pad + 1).astype(np.float32).reshape(k_count, 1)
        py = ed.node("Add", [ed.node("Add", [dy, gy], p), ed.const(p + "/ky", ky)], p)
        px = ed.node("Add", [ed.node("Add", [dx, gx], p), ed.const(p + "/kx", kx)], p)

        y0 = ed.node("Floor", [py], p)
        x0 = ed.node("Floor", [px], p)
        ly = ed.node("Sub", [py, y0], p)
        lx = ed.node("Sub", [px, x0], p)
        hy = ed.node("Sub", [one, ly], p)
        hx = ed.node("Sub", [one, lx], p)
        y1 = ed.node("Add", [y0, one], p)
        x1 = ed.node("Add", [x0, one], p)

        zf = zero
        ymax = ed.const(p + "/ymax", np.array(float(hp - 1), np.float32))
        xmax = ed.const(p + "/xmax", np.array(float(wp - 1), np.float32))
        y0c = ed.node("Clip", [y0, zf, ymax], p)
        y1c = ed.node("Clip", [y1, zf, ymax], p)
        x0c = ed.node("Clip", [x0, zf, xmax], p)
        x1c = ed.node("Clip", [x1, zf, xmax], p)
        wpf = ed.const(p + "/wp", np.array(float(wp), np.float32))
        row0 = ed.node("Mul", [y0c, wpf], p)
        row1 = ed.node("Mul", [y1c, wpf], p)

        m = ed.node("Reshape", [mask, ed.const(p + "/s", np.array([k_count, hw], np.int64))], p)
        hym = ed.node("Mul", [hy, m], p)
        lym = ed.node("Mul", [ly, m], p)

        corners = [
            (row0, x0c, hym, hx),
            (row0, x1c, hym, lx),
            (row1, x0c, lym, hx),
            (row1, x1c, lym, lx),
        ]
        # Flat sample index and weight of each bilinear corner, [K, HW] each.
        # int32 indices: the padded input has far fewer than 2^31 pixels.
        corner_idx, corner_w = [], []
        for row, col, wy, wx in corners:
            corner_idx.append(ed.node("Cast", [ed.node("Add", [row, col], p)], p, to=TensorProto.INT32))
            corner_w.append(ed.node("Mul", [wy, wx], p))

        # Taps are processed in groups; each corner is gathered separately so
        # that no intermediate is larger than GROUP_BYTES.
        per_tap_bytes = cin * hw * 4
        group = max(1, min(k_count, GROUP_BYTES // per_tap_bytes))
        axes0 = ed.const(p + "/ax0", np.array([0], np.int64))
        acc = None
        for k0 in range(0, k_count, group):
            g = min(group, k_count - k0)
            starts = ed.const(p + "/st", np.array([k0], np.int64))
            ends = ed.const(p + "/en", np.array([k0 + g], np.int64))
            flat = ed.const(p + "/s", np.array([g * hw], np.int64))
            row_shape = ed.const(p + "/s", np.array([1, g * hw], np.int64))
            sampled = None
            for idx_c, w_c in zip(corner_idx, corner_w):
                idx_g = ed.node("Reshape", [ed.node("Slice", [idx_c, starts, ends, axes0], p), flat], p)
                vals = ed.node("Gather", [xf, idx_g], p, axis=1)  # [C, g*HW]
                w_g = ed.node("Reshape", [ed.node("Slice", [w_c, starts, ends, axes0], p), row_shape], p)
                term = ed.node("Mul", [vals, w_g], p)
                sampled = term if sampled is None else ed.node("Add", [sampled, term], p)
            assert sampled is not None
            # [C, g*HW] is [C, g, HW] in memory, i.e. rows c*g + t as the weights below.
            sampled = ed.node("Reshape", [sampled, ed.const(p + "/s", np.array([cin * g, hw], np.int64))], p)
            # W_g[o, c*g + t] = W[o, c, ky(k0+t), kx(k0+t)]
            w_taps = weight.reshape(cout, cin, k_count)[:, :, k0 : k0 + g].reshape(cout, cin * g)
            contrib = ed.node("MatMul", [ed.const(p + "/weight", np.ascontiguousarray(w_taps)), sampled], p)
            acc = contrib if acc is None else ed.node("Add", [acc, contrib], p)
        assert acc is not None
        if bias is not None:
            acc = ed.node("Add", [acc, ed.const(p + "/bias", bias.reshape(cout, 1))], p)
        ed.node("Reshape", [acc, ed.const(p + "/s", np.array([1, cout, h, w], np.int64))], p, output=out_name)

        # Only the node producing the scope output is removed explicitly; the
        # rest of the old decomposition becomes dead code (see remove_unused).
        remove.add(id(final_add))

    kept = [n for n in graph.node if id(n) not in remove]
    del graph.node[:]
    graph.node.extend(kept + ed.new_nodes)
    graph.initializer.extend(ed.new_inits)
    return len(scopes)


def _const_value(graph: onnx.GraphProto, name: str):
    for t in graph.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    for n in graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            return numpy_helper.to_array(n.attribute[0].t)
    return None


def replace_patch_concats(model: onnx.ModelProto) -> int:
    """Rewrites the decoder's image-to-patches step as Reshape/Transpose.

    The export implements it with nested Splits and a Concat of up to 1024
    inputs (one per patch). WebGPU limits the number of buffers per shader
    (often 8–10), so that Concat cannot run on the GPU. The same reordering
    is expressed exactly by Reshape → Transpose → Reshape:
    out[(j·gh + i)·C + c, y, x] = img[c, i·ph + y, j·pw + x].
    """
    graph = model.graph
    shapes = infer_shapes(model)
    producer = {o: n for n in graph.node for o in n.output}
    ed = GraphEditor(model)
    remove: set[int] = set()
    count = 0
    for cat in graph.node:
        if cat.op_type != "Concat" or len(cat.input) < 8:
            continue
        unsq = [producer.get(i) for i in cat.input]
        if not all(u is not None and u.op_type == "Unsqueeze" for u in unsq):
            raise SystemExit(f"{cat.name}: unexpected large Concat layout")
        row_splits = [producer[u.input[0]] for u in unsq]
        col_split = producer[row_splits[0].input[0]]
        gw = len(col_split.output)
        gh = len(row_splits[0].output)
        if gw * gh != len(cat.input) or col_split.op_type != "Split":
            raise SystemExit(f"{cat.name}: unexpected patch grid")
        for k, (u, rs) in enumerate(zip(unsq, row_splits)):
            j, i = divmod(k, gh)
            if rs.input[0] != col_split.output[j] or rs.output[i] != u.input[0]:
                raise SystemExit(f"{cat.name}: patches are not in column-major order")
        axes = {a.name: helper.get_attribute_value(a) for a in col_split.attribute}
        row_axes = {a.name: helper.get_attribute_value(a) for a in row_splits[0].attribute}
        if axes.get("axis") != -1 or row_axes.get("axis") != -2:
            raise SystemExit(f"{cat.name}: unexpected split axes")
        cat_axis = {a.name: helper.get_attribute_value(a) for a in cat.attribute}.get("axis")
        unsq_axes = _const_value(graph, unsq[0].input[1])
        if cat_axis != 1 or unsq_axes is None or list(np.atleast_1d(unsq_axes)) != [0]:
            raise SystemExit(f"{cat.name}: unexpected concat axis")
        src = col_split.input[0]
        c, h, w = shapes[src]
        if h % gh or w % gw:
            raise SystemExit(f"{cat.name}: image not divisible into patches")
        ph, pw = h // gh, w // gw
        p = cat.name + "/patches"
        x = ed.node("Reshape", [src, ed.const(p + "/s", np.array([c, gh, ph, gw, pw], np.int64))], p)
        x = ed.node("Transpose", [x], p, perm=[3, 1, 0, 2, 4])
        ed.node("Reshape", [x, ed.const(p + "/s", np.array([1, gw * gh * c, ph, pw], np.int64))], p, output=cat.output[0])
        remove.add(id(cat))
        count += 1
    kept = [n for n in graph.node if id(n) not in remove]
    del graph.node[:]
    graph.node.extend(kept + ed.new_nodes)
    graph.initializer.extend(ed.new_inits)
    return count


def _attrs(node: onnx.NodeProto) -> dict:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    users: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for n in graph.node:
        for i in n.input:
            if i:
                users[i].append(n)
    for o in graph.output:
        users[o.name].append(None)  # type: ignore[arg-type]  # graph outputs count as a use
    return users


def _nbytes(shape: list[int] | None) -> int:
    return 4 * math.prod(shape) if shape else 0


def _conv_params(node: onnx.NodeProto, inits: dict[str, onnx.TensorProto]):
    """Weight and bias (float32) of a plain Conv (group 1, weights stored in the file), else None."""
    if node.op_type != "Conv" or node.input[1] not in inits or _attrs(node).get("group", 1) != 1:
        return None
    if len(node.input) > 2 and node.input[2] and node.input[2] not in inits:
        return None
    w = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32)
    b = numpy_helper.to_array(inits[node.input[2]]).astype(np.float32) if len(node.input) > 2 and node.input[2] else None
    return w, b


def _is_pointwise(node: onnx.NodeProto, weight: np.ndarray) -> bool:
    a = _attrs(node)
    return (
        weight.shape[2:] == (1, 1)
        and all(s == 1 for s in a.get("strides", [1, 1]))
        and not any(a.get("pads", [0, 0, 0, 0]))
        and a.get("auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET")
    )


def _replace_nodes(graph: onnx.GraphProto, ed: GraphEditor, remove: set[int]) -> None:
    kept = [n for n in graph.node if id(n) not in remove]
    del graph.node[:]
    graph.node.extend(kept + ed.new_nodes)
    graph.initializer.extend(ed.new_inits)


def _conv(ed: GraphEditor, x: str, w: np.ndarray, b: np.ndarray | None, like: onnx.NodeProto, prefix: str,
          output: str | None = None) -> str:
    """A Conv with the attributes of `like` and new weights."""
    attrs = {k: v for k, v in _attrs(like).items() if k != "kernel_shape"}
    inputs = [x, ed.const(prefix + "/w", np.ascontiguousarray(w, np.float32))]
    if b is not None:
        inputs.append(ed.const(prefix + "/b", np.ascontiguousarray(b, np.float32)))
    return ed.node("Conv", inputs, prefix, output=output, kernel_shape=list(w.shape[2:]), **attrs)


def _sum(ed: GraphEditor, parts: list[str], prefix: str, output: str) -> None:
    acc = parts[0]
    for k, part in enumerate(parts[1:], 1):
        acc = ed.node("Add", [acc, part], prefix, output=output if k == len(parts) - 1 else None)
    if len(parts) == 1:
        ed.node("Identity", [acc], prefix, output=output)


def split_concat_convs(model: onnx.ModelProto, min_bytes: int) -> int:
    """Conv1x1(Concat(x₁ … xₙ)) → Conv1x1(x₁, W₁) + … + Conv1x1(xₙ, Wₙ).

    A pointwise convolution over concatenated channels is the sum of pointwise
    convolutions over each part, so the (large) concatenated tensor is never
    materialised and each part can be released as soon as its term is added.
    """
    graph = model.graph
    shapes = infer_shapes(model)
    inits = {t.name: t for t in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    users = _consumers(graph)
    ed = GraphEditor(model)
    remove: set[int] = set()
    for conv in graph.node:
        params = _conv_params(conv, inits)
        cat = producer.get(conv.input[0]) if params else None
        if params is None or not _is_pointwise(conv, params[0]) or cat is None or cat.op_type != "Concat":
            continue
        if _attrs(cat).get("axis") != 1 or len(users[cat.output[0]]) != 1:
            continue
        if _nbytes(shapes.get(cat.output[0])) < min_bytes:
            continue
        w, b = params
        p = conv.name + "/split"
        parts, start = [], 0
        for k, x in enumerate(cat.input):
            c = shapes[x][1]
            parts.append(_conv(ed, x, w[:, start : start + c], b if k == 0 else None, conv, p))
            start += c
        if start != w.shape[1]:
            raise SystemExit(f"{conv.name}: channel count mismatch")
        _sum(ed, parts, p, conv.output[0])
        remove.update({id(conv), id(cat)})
    _replace_nodes(graph, ed, remove)
    return len(remove) // 2


def move_convs_before_resizes(model: onnx.ModelProto) -> int:
    """Conv1x1(Resize(x)) → Resize(Conv1x1(x)) when the convolution reduces channels.

    Linear (and nearest) resizing mixes pixels with weights that sum to one and
    a pointwise convolution mixes channels per pixel, so the two commute
    (bias included). The convolution then runs at the low resolution and the
    wide upsampled tensor is never created.
    """
    graph = model.graph
    shapes = infer_shapes(model)
    inits = {t.name: t for t in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    users = _consumers(graph)
    ed = GraphEditor(model)
    remove: set[int] = set()
    for conv in graph.node:
        params = _conv_params(conv, inits)
        rs = producer.get(conv.input[0]) if params else None
        if params is None or not _is_pointwise(conv, params[0]) or rs is None or rs.op_type != "Resize":
            continue
        a = _attrs(rs)
        if a.get("mode", b"nearest") not in (b"linear", b"nearest") or a.get("antialias", 0):
            continue
        if len(users[rs.output[0]]) != 1:
            continue
        x, out_shape = rs.input[0], shapes.get(rs.output[0])
        w, b = params
        if out_shape is None or x not in shapes or w.shape[0] >= shapes[x][1]:
            continue
        p = conv.name + "/before_resize"
        y = _conv(ed, x, w, b, conv, p)
        inputs = [y] + list(rs.input[1:])
        if len(inputs) > 3 and inputs[3]:  # sizes: the channel count changes
            inputs[3] = ed.const(p + "/sizes", np.array([out_shape[0], w.shape[0], *out_shape[2:]], np.int64))
        ed.new_nodes.append(helper.make_node("Resize", inputs, [conv.output[0]], name=ed.name(p + "/Resize"), **a))
        remove.update({id(conv), id(rs)})
    _replace_nodes(graph, ed, remove)
    return len(remove) // 2


def fold_pointwise_convs(model: onnx.ModelProto) -> int:
    """Conv1x1(Conv(x, W₁, b₁), W₂, b₂) → Conv(x, W₂·W₁, W₂·b₁ + b₂).

    Two linear maps without an activation in between are one linear map; the
    pointwise convolution has no padding, so this is exact at the borders too.
    Only applied when it reduces channels (the intermediate tensor disappears).
    """
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    users = _consumers(graph)
    ed = GraphEditor(model)
    remove: set[int] = set()
    for conv in graph.node:
        outer = _conv_params(conv, inits)
        first = producer.get(conv.input[0]) if outer else None
        if outer is None or not _is_pointwise(conv, outer[0]) or first is None or id(first) in remove:
            continue
        inner = _conv_params(first, inits)
        if inner is None or len(users[first.output[0]]) != 1 or outer[0].shape[0] >= inner[0].shape[0]:
            continue
        w2 = outer[0][:, :, 0, 0].astype(np.float64)
        w1, b1 = inner[0].astype(np.float64), inner[1]
        w = np.einsum("oc,cikl->oikl", w2, w1)
        b = np.zeros(w.shape[0])
        if b1 is not None:
            b += w2 @ b1.astype(np.float64)
        if outer[1] is not None:
            b += outer[1]
        _conv(ed, first.input[0], w, b, first, conv.name + "/folded", output=conv.output[0])
        remove.update({id(conv), id(first)})
    _replace_nodes(graph, ed, remove)
    return len(remove) // 2


def chunk_wide_convs(model: onnx.ModelProto, max_bytes: int) -> int:
    """Conv_B(Conv_A(x)) with a huge intermediate → Σₖ Conv_B,k(Conv_A,k(x)).

    Conv_A's output channels are computed in chunks; each chunk goes straight
    into its share of Conv_B (a convolution is a sum over input channels), so
    only one chunk of the intermediate tensor exists at a time.
    """
    graph = model.graph
    shapes = infer_shapes(model)
    inits = {t.name: t for t in graph.initializer}
    users = _consumers(graph)
    ed = GraphEditor(model)
    remove: set[int] = set()
    for first in graph.node:
        inner = _conv_params(first, inits)
        size = _nbytes(shapes.get(first.output[0]))
        if inner is None or size <= max_bytes or len(users[first.output[0]]) != 1:
            continue
        second = users[first.output[0]][0]
        outer = _conv_params(second, inits) if second is not None else None
        if outer is None or second.input[0] != first.output[0] or {id(first), id(second)} & remove:
            continue
        channels = inner[0].shape[0]
        chunks = math.ceil(size / max_bytes)
        while channels % chunks:
            chunks += 1
        step = channels // chunks
        p = second.name + "/chunked"
        parts = []
        for k in range(chunks):
            s = slice(k * step, (k + 1) * step)
            y = _conv(ed, first.input[0], inner[0][s], inner[1][s] if inner[1] is not None else None, first, p)
            parts.append(_conv(ed, y, outer[0][:, s], outer[1] if k == 0 else None, second, p))
        _sum(ed, parts, p, second.output[0])
        remove.update({id(first), id(second)})
    _replace_nodes(graph, ed, remove)
    return len(remove) // 2


def check_buffer_counts(model: onnx.ModelProto, limit: int = 8) -> list[str]:
    """Nodes that would need more storage buffers than WebGPU guarantees by default."""
    cpu_ops = {"Shape", "Constant", "ConstantOfShape", "Range"}
    return [
        f"{n.op_type} {n.name} ({len([i for i in n.input if i])} in, {len(n.output)} out)"
        for n in model.graph.node
        if n.op_type not in cpu_ops and len([i for i in n.input if i]) + len(n.output) > limit
    ]


def add_sigmoid_output(model: onnx.ModelProto) -> None:
    graph = model.graph
    assert len(graph.output) == 1
    logits = graph.output[0].name
    for n in graph.node:
        n.output[:] = [("logits" if o == logits else o) for o in n.output]
    graph.node.append(helper.make_node("Sigmoid", ["logits"], [OUTPUT_NAME], name="alpha/Sigmoid"))
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, 1, INPUT_SIZE, INPUT_SIZE])
    )


def remove_unused(model: onnx.ModelProto) -> None:
    graph = model.graph
    used = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
    # Drop nodes whose outputs are never used (iteratively).
    changed = True
    while changed:
        changed = False
        keep = []
        for n in graph.node:
            if any(o in used for o in n.output):
                keep.append(n)
            else:
                changed = True
        if changed:
            del graph.node[:]
            graph.node.extend(keep)
            used = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
    inits = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)


def topological_sort(model: onnx.ModelProto) -> None:
    graph = model.graph
    available = {i.name for i in graph.input} | {t.name for t in graph.initializer} | {""}
    producers = {}
    for idx, n in enumerate(graph.node):
        for o in n.output:
            producers[o] = idx
    deps: list[set[int]] = []
    users: dict[int, list[int]] = defaultdict(list)
    for idx, n in enumerate(graph.node):
        d = {producers[i] for i in n.input if i not in available and i in producers}
        deps.append(d)
        for p in d:
            users[p].append(idx)
    # Depth-first ordering keeps each chain of the deformable-conv accumulation
    # contiguous, which keeps peak memory low when executed in order.
    order: list[int] = []
    state = [0] * len(graph.node)  # 0 = new, 1 = visiting, 2 = done
    output_producers = [producers[o.name] for o in graph.output]
    stack: list[tuple[int, bool]] = [(i, False) for i in reversed(output_producers)]
    while stack:
        idx, expanded = stack.pop()
        if expanded:
            if state[idx] != 2:
                state[idx] = 2
                order.append(idx)
            continue
        if state[idx] != 0:
            continue
        state[idx] = 1
        stack.append((idx, True))
        node_inputs = [producers[i] for i in graph.node[idx].input if i in producers and i not in available]
        for dep in reversed(node_inputs):
            if state[dep] == 0:
                stack.append((dep, False))
    nodes = [graph.node[i] for i in order]
    del graph.node[:]
    graph.node.extend(nodes)


def _quant_axis(graph: onnx.GraphProto, name: str) -> int | None:
    """Output-channel axis for per-channel quantisation, or None if unsuitable."""
    axes = set()
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp != name:
                continue
            if n.op_type == "Conv" and i == 1:
                axes.add(0)
            elif n.op_type == "MatMul" and i == 1:
                axes.add(-1)
            elif n.op_type == "MatMul" and i == 0:
                axes.add(0)
            elif n.op_type == "Identity":
                axes.add(-1)
            else:
                return None
    return axes.pop() if len(axes) == 1 else None


def compress_weights(model: onnx.ModelProto, mode: str) -> dict[str, int]:
    graph = model.graph
    stats = {"compressed": 0, "kept": 0}
    if mode == "fp32":
        return stats
    new_inits, cast_nodes = [], []
    for t in graph.initializer:
        n_el = int(np.prod(t.dims)) if t.dims else 1
        if t.data_type != TensorProto.FLOAT or n_el < MIN_COMPRESS_ELEMENTS:
            new_inits.append(t)
            stats["kept"] += 1
            continue
        arr = numpy_helper.to_array(t)
        if mode == "fp16":
            if np.abs(arr).max() > 60000:
                raise SystemExit(f"{t.name}: value out of float16 range")
            stored = numpy_helper.from_array(arr.astype(np.float16), t.name + "::fp16")
            new_inits.append(stored)
            cast_nodes.append(
                helper.make_node("Cast", [stored.name], [t.name], name=t.name + "::cast", to=TensorProto.FLOAT)
            )
        elif mode == "int8":
            axis = _quant_axis(graph, t.name)
            if axis is None:
                stored = numpy_helper.from_array(arr.astype(np.float16), t.name + "::fp16")
                new_inits.append(stored)
                cast_nodes.append(
                    helper.make_node("Cast", [stored.name], [t.name], name=t.name + "::cast", to=TensorProto.FLOAT)
                )
            else:
                red = tuple(i for i in range(arr.ndim) if i != axis % arr.ndim)
                scale = np.abs(arr).max(axis=red, keepdims=True) / 127.0
                scale[scale == 0] = 1.0
                q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
                stored = numpy_helper.from_array(q, t.name + "::q")
                sc = numpy_helper.from_array(scale.astype(np.float32), t.name + "::scale")
                new_inits += [stored, sc]
                cast_nodes.append(
                    helper.make_node("Cast", [stored.name], [t.name + "::deq"], name=t.name + "::cast", to=TensorProto.FLOAT)
                )
                cast_nodes.append(
                    helper.make_node("Mul", [t.name + "::deq", sc.name], [t.name], name=t.name + "::mul")
                )
        else:
            raise SystemExit(f"unknown weight mode {mode}")
        stats["compressed"] += 1
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(cast_nodes + nodes)
    return stats


def write_chunks(model_bytes: bytes, out_dir: Path, chunk_mib: int, version: str, weights: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("model.onnx.part*"):
        old.unlink()
    chunk = chunk_mib * 1024 * 1024
    parts = []
    for i in range(math.ceil(len(model_bytes) / chunk)):
        data = model_bytes[i * chunk : (i + 1) * chunk]
        name = f"model.onnx.part{i:02d}"
        (out_dir / name).write_bytes(data)
        parts.append({"file": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "id": f"birefnet-lite-{version}",
        "version": version,
        "name": SOURCE["name"],
        "task": "general background removal (dichotomous segmentation)",
        "input": {"name": INPUT_NAME, "width": INPUT_SIZE, "height": INPUT_SIZE,
                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"name": OUTPUT_NAME, "width": INPUT_SIZE, "height": INPUT_SIZE},
        "weights": weights,
        "size": len(model_bytes),
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "parts": parts,
        "source": SOURCE,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "LICENSE.txt").write_text(
        "The model files in this directory are derived from BiRefNet\n"
        f"({SOURCE['homepage']}), released under the MIT License:\n\n" + MIT_LICENSE_TEXT
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path(".cache/model-source"), help="download cache directory")
    ap.add_argument("--source", type=Path, help="use an already downloaded source ONNX file")
    ap.add_argument("--out", type=Path, default=Path("public/models/birefnet-lite"), help="output directory")
    ap.add_argument("--weights", choices=["fp16", "fp32", "int8"], default="fp16", help="weight storage precision")
    ap.add_argument("--chunk-mib", type=int, default=24, help="chunk size in MiB")
    ap.add_argument("--version", default="1", help="model build version (bump to invalidate caches)")
    ap.add_argument("--save-onnx", type=Path, help="additionally write the unsplit model here")
    ap.add_argument(
        "--verify-output",
        action="store_true",
        help="fail unless the result is byte-identical to the validated model",
    )
    args = ap.parse_args()

    source = args.source or fetch_source(args.cache)
    if args.source:
        digest = sha256_file(source)
        if digest != SOURCE["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {source}: {digest}")
    model = onnx.load(str(source))
    n = replace_deformable_convs(model)
    log(f"Replaced {n} deformable convolutions")
    n = replace_patch_concats(model)
    log(f"Rewrote {n} patch concatenations")
    # Shape inference needs the nodes in topological order, so tidy up after every pass.
    counts = []
    for rewrite in (
        lambda m: split_concat_convs(m, SPLIT_MIN_BYTES),
        move_convs_before_resizes,
        fold_pointwise_convs,
        lambda m: chunk_wide_convs(m, CHUNK_MAX_BYTES),
    ):
        remove_unused(model)
        topological_sort(model)
        counts.append(rewrite(model))
    log("Memory rewrites: {} concat convs split, {} convs moved before resizes, {} folded, {} chunked".format(*counts))
    add_sigmoid_output(model)
    remove_unused(model)
    topological_sort(model)
    too_wide = check_buffer_counts(model)
    if too_wide:
        raise SystemExit("Nodes exceed the WebGPU buffer limit:\n  " + "\n  ".join(too_wide))
    stats = compress_weights(model, args.weights)
    log(f"Weights: {stats}")
    model.producer_name = "cutout/convert_birefnet.py"
    model.doc_string = (
        f"Derived from {SOURCE['url']} (sha256 {SOURCE['sha256']}). "
        f"{SOURCE['license']} License, {SOURCE['copyright']}."
    )
    onnx.checker.check_model(model)
    data = model.SerializeToString()
    if args.save_onnx:
        args.save_onnx.parent.mkdir(parents=True, exist_ok=True)
        args.save_onnx.write_bytes(data)
    manifest = write_chunks(data, args.out, args.chunk_mib, args.version, args.weights)
    log(f"Wrote {len(manifest['parts'])} parts, {manifest['size'] / 1e6:.1f} MB, sha256 {manifest['sha256']}")
    if args.verify_output:
        expected = VALIDATED_OUTPUT_SHA256.get(args.weights)
        if manifest["sha256"] != expected:
            raise SystemExit(
                f"Converted model differs from the validated one ({expected}). "
                "Check tool versions (tools/model/requirements.txt) or re-validate and update the hash."
            )
        log("Output matches the validated model")


if __name__ == "__main__":
    os.umask(0o022)
    main()

import torch
import onnx
from onnx import version_converter
from onnx2torch import convert

BASE = "/home/arduino/yolo_test"

original = onnx.load(f"{BASE}/yolov11_det.onnx")
step1 = version_converter.convert_version(original, 18)
# Split-18 nodes in this graph don't use num_outputs (verified: only 'axis' attr,
# split sizes passed as 2nd input) so they are byte-compatible with Split-13.
# The official version_converter has no adapter for Split 18->13, so relabel
# the opset metadata directly instead of a semantic conversion.
for op in step1.opset_import:
    if op.domain in ("", "ai.onnx"):
        op.version = 13

# opset18 moved Reduce* axes from an attribute to an optional 2nd input.
# onnx2torch's Reduce converter (registered for the older, pre-18 signature)
# expects axes as an attribute, so fold any constant-initializer axes input
# back into an attribute for nodes that still have the newer 2-input form.
from onnx import numpy_helper, helper

initializer_map = {init.name: init for init in step1.graph.initializer}
REDUCE_OPS = {
    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
}
for n in step1.graph.node:
    if n.op_type in REDUCE_OPS and len(n.input) == 2:
        axes_name = n.input[1]
        if axes_name in initializer_map:
            axes_val = numpy_helper.to_array(initializer_map[axes_name]).tolist()
            del n.input[1]
            n.attribute.append(helper.make_attribute("axes", axes_val))
            print(f"Patched {n.op_type} ({n.name}): folded axes input -> attribute {axes_val}")

onnx.save(step1, f"{BASE}/yolov11_det_op13.onnx")

onnx_model = convert(f"{BASE}/yolov11_det_op13.onnx")
onnx_model.eval()

example_input = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    traced = torch.jit.trace(onnx_model, example_input)

traced.save(f"{BASE}/yolov11_det.pt")
print("Saved yolov11_det.pt")

with torch.no_grad():
    out = traced(example_input)
if isinstance(out, (tuple, list)):
    for i, o in enumerate(out):
        print(f"output {i} shape:", o.shape)
else:
    print("output shape:", out.shape)

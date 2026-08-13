import sys
import torch
import onnx
from onnx import version_converter
from onnx2torch import convert

BASE = "/home/arduino/yolo_test"
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 16

original = onnx.load(f"{BASE}/yolov11_det.onnx")
step1 = version_converter.convert_version(original, 18)
for op in step1.opset_import:
    if op.domain in ("", "ai.onnx"):
        op.version = 13

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

# YOLOv11's attention block (C2PSA) and detection head bake batch=1 into
# Reshape target-shape constants. Rewrite dim0 -> BATCH so tracing at a
# different batch size doesn't hit a hardcoded-shape mismatch.
for n in step1.graph.node:
    if n.op_type == "Reshape" and len(n.input) == 2:
        shp_name = n.input[1]
        if shp_name in initializer_map:
            init = initializer_map[shp_name]
            val = numpy_helper.to_array(init).copy()
            if len(val) > 0 and val[0] == 1:
                val[0] = BATCH
                new_init = numpy_helper.from_array(val, name=init.name)
                init.CopyFrom(new_init)
                print(f"Patched reshape target for {n.name}: dim0 1 -> {BATCH}")

onnx.save(step1, f"{BASE}/yolov11_det_op13.onnx")

full_model = convert(f"{BASE}/yolov11_det_op13.onnx")
full_model.eval()


class BoxesOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0]  # boxes only


wrapper = BoxesOnly(full_model)

example_input = torch.rand(BATCH, 3, 640, 640)
with torch.no_grad():
    traced = torch.jit.trace(wrapper, example_input)

traced.save(f"{BASE}/yolov11_det_b{BATCH}.pt")
print(f"Saved yolov11_det_b{BATCH}.pt")

with torch.no_grad():
    out = traced(example_input)
print("boxes output shape:", out.shape)

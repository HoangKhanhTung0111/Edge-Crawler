import torch
import onnx
from onnx import version_converter
from onnx2torch import convert

BASE = "/home/arduino/gpu_test"

original = onnx.load(f"{BASE}/mobilenet_v2.onnx")
downgraded = version_converter.convert_version(original, 17)
onnx.save(downgraded, f"{BASE}/mobilenet_v2_op13.onnx")

onnx_model = convert(f"{BASE}/mobilenet_v2_op13.onnx")
onnx_model.eval()

example_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    traced = torch.jit.trace(onnx_model, example_input)

traced.save(f"{BASE}/mobilenet_v2.pt")
print("Saved mobilenet_v2.pt")

with torch.no_grad():
    out = traced(example_input)
print("Output shape:", out.shape)

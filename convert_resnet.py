import torch
import onnx
from onnx2torch import convert

BASE = "/home/arduino/resnet_test"
BATCH = 8

try:
    onnx_model = convert(f"{BASE}/resnet50.onnx")
except Exception as e:
    print("Direct convert failed, trying opset downgrade:", e)
    from onnx import version_converter
    original = onnx.load(f"{BASE}/resnet50.onnx")
    downgraded = version_converter.convert_version(original, 17)
    onnx.save(downgraded, f"{BASE}/resnet50_op13.onnx")
    onnx_model = convert(f"{BASE}/resnet50_op13.onnx")

onnx_model.eval()

example_input = torch.rand(BATCH, 3, 224, 224)
with torch.no_grad():
    traced = torch.jit.trace(onnx_model, example_input)

traced.save(f"{BASE}/resnet50_b{BATCH}.pt")
print(f"Saved resnet50_b{BATCH}.pt")

with torch.no_grad():
    out = traced(example_input)
print("output shape:", out.shape)

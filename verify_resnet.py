import numpy as np
from PIL import Image
import onnxruntime as ort
import torch

BASE = "/home/arduino/resnet_test"


def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # value_range [0,1] per metadata; norm is inside the graph
    return np.transpose(arr, (2, 0, 1))


single = preprocess(f"{BASE}/../gpu_test/test_image.jpg")
batch_input = np.ascontiguousarray(np.stack([single] * 8, axis=0))

sess = ort.InferenceSession(f"{BASE}/resnet50.onnx", providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {"image_tensor": batch_input})[0]
print("ONNX top1 idx (img0):", onnx_out[0].argmax(), "score:", onnx_out[0].max())

traced = torch.jit.load(f"{BASE}/resnet50_b8.pt")
traced.eval()
with torch.no_grad():
    t_out = traced(torch.from_numpy(batch_input)).numpy()
print("Torch top1 idx (img0):", t_out[0].argmax(), "score:", t_out[0].max())

diff = np.abs(onnx_out - t_out).max()
print("max diff:", diff)

with open(f"{BASE}/labels.txt") as f:
    labels = [l.strip() for l in f.readlines()]
idx = onnx_out[0].argmax()
print("label:", labels[idx] if idx < len(labels) else "unknown")

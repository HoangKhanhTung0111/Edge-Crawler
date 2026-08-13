import numpy as np
from PIL import Image
import onnxruntime as ort
import torch

BASE = "/home/arduino/yolo_test"


def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((640, 640), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


input_data = preprocess(f"{BASE}/../gpu_test/test_image.jpg")

# ONNX ground truth (original, unmodified opset21 model)
sess = ort.InferenceSession(f"{BASE}/yolov11_det.onnx", providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {"image": input_data})
onnx_boxes, onnx_scores, onnx_class = onnx_out
print("ONNX top score:", onnx_scores.max(), "argmax idx:", onnx_scores.argmax())
top_i = onnx_scores.argmax()
print("ONNX top box:", onnx_boxes[0, top_i], "class:", onnx_class[0, top_i], "score:", onnx_scores[0, top_i])

# count boxes above threshold
thresh = 0.25
n_above = (onnx_scores[0] > thresh).sum()
print(f"ONNX: {n_above} boxes above {thresh}")

# TorchScript (converted) output
traced = torch.jit.load(f"{BASE}/yolov11_det.pt")
traced.eval()
with torch.no_grad():
    t_boxes, t_scores, t_class = traced(torch.from_numpy(input_data))
t_scores = t_scores.numpy()
t_boxes = t_boxes.numpy()
t_class = t_class.numpy()
print("Torch top score:", t_scores.max(), "argmax idx:", t_scores.argmax())
top_i_t = t_scores.argmax()
print("Torch top box:", t_boxes[0, top_i_t], "class:", t_class[0, top_i_t], "score:", t_scores[0, top_i_t])
n_above_t = (t_scores[0] > thresh).sum()
print(f"Torch: {n_above_t} boxes above {thresh}")

# direct diff
box_diff = np.abs(onnx_boxes - t_boxes).max()
score_diff = np.abs(onnx_scores - t_scores).max()
print(f"max box diff: {box_diff:.6f}, max score diff: {score_diff:.6f}")

import torch
import numpy as np
from PIL import Image

BASE = "/home/arduino/gpu_test"


def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


with open(f"{BASE}/labels.txt") as f:
    labels = [l.strip() for l in f.readlines()]

traced = torch.jit.load(f"{BASE}/mobilenet_v2.pt")
traced.eval()

input_data = preprocess(f"{BASE}/test_image.jpg")
with torch.no_grad():
    out = traced(torch.from_numpy(input_data))

output_data = out.squeeze().numpy()
exp_scores = np.exp(output_data - np.max(output_data))
probabilities = exp_scores / np.sum(exp_scores)
top_5 = np.argsort(probabilities)[-5:][::-1]

for i, idx in enumerate(top_5):
    name = labels[idx] if idx < len(labels) else "Unknown"
    print(f"{i+1}. {name} (Index: {idx}) - {probabilities[idx]*100:.2f}%")

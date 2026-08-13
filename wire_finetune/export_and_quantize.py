import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import qai_hub as hub

BASE = Path(r"C:\Users\Admin\unoq\wire_finetune")
RUNS = BASE / "runs"
DATASET = BASE / "dataset"

# --- Rebuild model architecture and load finetuned weights ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
state = torch.load(RUNS / "mobilenet_v2_wire_best.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

example_input = torch.rand(1, 3, 224, 224)
with torch.no_grad():
    out = model(example_input)
print("Sanity forward pass output shape:", out.shape)

onnx_path = RUNS / "wire_classifier.onnx"
torch.onnx.export(
    model, example_input, str(onnx_path),
    input_names=["image_tensor"], output_names=["class_logits"],
    opset_version=17,
    dynamic_axes=None,
)
print(f"Saved ONNX -> {onnx_path}")

# --- Build calibration data from real training images (same preprocessing, no augmentation) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
calib_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

all_images = list((DATASET / "intact").rglob("*.jpg")) + list((DATASET / "broken").rglob("*.jpg"))
random.seed(0)
random.shuffle(all_images)
calib_images = all_images[:100]  # 100 samples is plenty for PTQ calibration

calib_tensors = []
for p in calib_images:
    img = Image.open(p).convert("RGB")
    t = calib_transform(img).unsqueeze(0).numpy().astype(np.float32)
    calib_tensors.append(t)

calibration_data = {"image_tensor": calib_tensors}
print(f"Calibration data: {len(calib_tensors)} anh")

# --- Submit quantize job (INT8 weights + activations, matches proven w8a8 TFLite/XNNPack recipe) ---
print("Submitting quantize job...")
qjob = hub.submit_quantize_job(
    str(onnx_path),
    calibration_data=calibration_data,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
    name="wire_classifier_w8a8",
)
print(f"Quantize job: {qjob.job_id} -> {qjob.url}")
qjob.wait()
print("Quantize status:", qjob.get_status())

quantized_model = qjob.get_target_model()
print("Got quantized model:", quantized_model)

# --- Compile quantized ONNX -> TFLite for the target device ---
device = hub.Device("Snapdragon 8 Elite QRD")  # closest available device w/ similar Hexagon; runtime target matters more than exact device for tflite CPU export
print("Submitting compile job (tflite)...")
cjob = hub.submit_compile_job(
    quantized_model,
    device=device,
    name="wire_classifier_tflite_w8a8",
    options="--target_runtime tflite",
)
print(f"Compile job: {cjob.job_id} -> {cjob.url}")
cjob.wait()
print("Compile status:", cjob.get_status())

target_model = cjob.get_target_model()
out_path = RUNS / "wire_classifier_quantized.tflite"
target_model.download(str(out_path))
print(f"Downloaded quantized tflite -> {out_path}")

with open(RUNS / "quantize_job_info.json", "w") as f:
    json.dump({
        "quantize_job_id": qjob.job_id,
        "quantize_job_url": qjob.url,
        "compile_job_id": cjob.job_id,
        "compile_job_url": cjob.url,
    }, f, indent=2)

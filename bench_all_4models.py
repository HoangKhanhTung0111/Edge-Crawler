import json
import time
import numpy as np
from PIL import Image
import onnxruntime as ort

IMAGE_PATH = "/home/arduino/gpu_test/test_image.jpg"
N_ITERS = 20


def preprocess(path, size, dtype, quant_params):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    img = img.crop((l, t, l + m, t + m)).resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # value_range [0,1]
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, 0)

    if dtype == "float32":
        return arr.astype(np.float32)

    scale = quant_params["scale"]
    zero_point = quant_params["zero_point"]
    q = arr / scale + zero_point
    np_dtype = {"uint8": np.uint8, "uint16": np.uint16, "int8": np.int8, "int16": np.int16}[dtype]
    info = np.iinfo(np_dtype)
    q = np.clip(q, info.min, info.max).astype(np_dtype)
    return q


def bench_onnx_model(model_dir, label):
    with open(f"{model_dir}/metadata.json") as f:
        meta = json.load(f)
    model_file = list(meta["model_files"].keys())[0]
    inputs_meta = meta["model_files"][model_file]["inputs"]
    input_name = list(inputs_meta.keys())[0]
    input_info = inputs_meta[input_name]
    size = input_info["shape"][2]
    dtype = input_info["dtype"]
    quant_params = input_info.get("quantization_parameters", {})

    data = preprocess(IMAGE_PATH, size, dtype, quant_params)

    sess = ort.InferenceSession(f"{model_dir}/{model_file}", providers=['CPUExecutionProvider'])

    sess.run(None, {input_name: data})  # warm-up
    t0 = time.time()
    for _ in range(N_ITERS):
        out = sess.run(None, {input_name: data})
    dt = (time.time() - t0) * 1000 / N_ITERS

    # decode top1 for a sanity check (best-effort, dequantize output if needed)
    out0 = out[0]
    out_meta = list(meta["model_files"][model_file]["outputs"].values())[0]
    if out_meta["dtype"] != "float32" and "quantization_parameters" in out_meta:
        oscale = out_meta["quantization_parameters"]["scale"]
        ozp = out_meta["quantization_parameters"]["zero_point"]
        out0 = (out0.astype(np.float32) - ozp) * oscale
    top1 = int(np.squeeze(out0).argmax())

    print(f"{label}: {dt:.2f} ms/anh  (input {size}x{size}, dtype={dtype}, top1_idx={top1})")
    return dt


results = {}

print("=" * 70)
print("MobileNet-v3-Small")
print("=" * 70)
results['mv3s_float'] = bench_onnx_model("/home/arduino/bench_models/mv3s_float", "  Float (ONNX Runtime)")
results['mv3s_quant'] = bench_onnx_model("/home/arduino/bench_models/mv3s_quant", "  Quantized w8a16 (ONNX Runtime)")

print("=" * 70)
print("EfficientNet-B0")
print("=" * 70)
results['effb0_float'] = bench_onnx_model("/home/arduino/bench_models/effb0_float", "  Float (ONNX Runtime)")
results['effb0_quant'] = bench_onnx_model("/home/arduino/bench_models/effb0_quant", "  Quantized w8a16 (ONNX Runtime)")

print("=" * 70)
print("EfficientNet-V2-S")
print("=" * 70)
results['effv2s_float'] = bench_onnx_model("/home/arduino/bench_models/effv2s_float", "  Float (ONNX Runtime)")
results['effv2s_quant'] = bench_onnx_model("/home/arduino/bench_models/effv2s_quant", "  Quantized w8a16 (ONNX Runtime)")

print()
print("=" * 70)
print("TONG HOP")
print("=" * 70)
print(f"{'Model':25} | {'Float (ms)':>12} | {'Quantized (ms)':>15} | {'Ty le':>8}")
print("-" * 70)
for name, fkey, qkey in [("MobileNet-v3-Small", 'mv3s_float', 'mv3s_quant'),
                          ("EfficientNet-B0", 'effb0_float', 'effb0_quant'),
                          ("EfficientNet-V2-S", 'effv2s_float', 'effv2s_quant')]:
    f_ms, q_ms = results[fkey], results[qkey]
    print(f"{name:25} | {f_ms:12.2f} | {q_ms:15.2f} | {f_ms/q_ms:7.2f}x")

with open("/home/arduino/bench_models/results.json", "w") as f:
    json.dump(results, f, indent=2)

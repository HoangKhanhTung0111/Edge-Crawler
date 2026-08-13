import time
import numpy as np
from PIL import Image
import ai_edge_litert.interpreter as tflite

IMAGE_PATH = "/home/arduino/gpu_test/test_image.jpg"
N_ITERS = 20


def preprocess(path, size, input_details):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    img = img.crop((l, t, l + m, t + m)).resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # NHWC, value_range [0,1]
    arr = np.expand_dims(arr, 0)

    dtype = input_details['dtype']
    if dtype == np.float32:
        return arr.astype(np.float32)

    scale, zero_point = input_details.get('quantization', (0.0, 0))
    q = arr / scale + zero_point
    info = np.iinfo(dtype)
    q = np.clip(q, info.min, info.max).astype(dtype)
    return q


def bench(model_path, label):
    interp = tflite.Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    size = inp['shape'][1]

    data = preprocess(IMAGE_PATH, size, inp)
    interp.set_tensor(inp['index'], data)
    interp.invoke()  # warm-up

    t0 = time.time()
    for _ in range(N_ITERS):
        interp.set_tensor(inp['index'], data)
        interp.invoke()
        output_data = interp.get_tensor(out['index'])
    dt = (time.time() - t0) * 1000 / N_ITERS

    if out['dtype'] != np.float32:
        oscale, ozp = out.get('quantization', (0.0, 0))
        output_data = (output_data.astype(np.float32) - ozp) * oscale
    top1 = int(np.squeeze(output_data).argmax())

    print(f"{label}: {dt:.2f} ms/anh  (dtype={inp['dtype'].__name__}, top1_idx={top1})")
    return dt


BASE = "/home/arduino/bench_tflite"
results = {}

print("=" * 60)
print("ResNet18")
print("=" * 60)
results['resnet18_float'] = bench(f"{BASE}/resnet18_float.tflite", "  Float")
results['resnet18_quant'] = bench(f"{BASE}/resnet18_quant.tflite", "  Quantized w8a8")

print("=" * 60)
print("SqueezeNet1.1")
print("=" * 60)
results['squeezenet_float'] = bench(f"{BASE}/squeezenet_float.tflite", "  Float")
results['squeezenet_quant'] = bench(f"{BASE}/squeezenet_quant.tflite", "  Quantized w8a8")

print("=" * 60)
print("ShuffleNet-v2")
print("=" * 60)
results['shufflenet_float'] = bench(f"{BASE}/shufflenet_float.tflite", "  Float")
results['shufflenet_quant'] = bench(f"{BASE}/shufflenet_quant.tflite", "  Quantized w8a8")

print()
print("=" * 60)
print("TONG HOP (TFLite + XNNPack, num_threads=4)")
print("=" * 60)
print(f"{'Model':20} | {'Float (ms)':>12} | {'Quantized (ms)':>15} | {'Ty le':>8}")
print("-" * 62)
for name, fkey, qkey in [("ResNet18", 'resnet18_float', 'resnet18_quant'),
                          ("SqueezeNet1.1", 'squeezenet_float', 'squeezenet_quant'),
                          ("ShuffleNet-v2", 'shufflenet_float', 'shufflenet_quant')]:
    f_ms, q_ms = results[fkey], results[qkey]
    print(f"{name:20} | {f_ms:12.2f} | {q_ms:15.2f} | {f_ms/q_ms:7.2f}x")

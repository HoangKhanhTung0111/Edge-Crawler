import os
import time
import threading
import numpy as np
from PIL import Image

DURATION_SEC = 5.0
GPU_TEST_DIR = "/home/arduino/gpu_test"

# ---------- CPU path: TFLite quantized (proven ~24ms via XNNPACK, num_threads=4) ----------
import ai_edge_litert.interpreter as tflite


def cpu_preprocess(path, input_details):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    img = img.crop((l, t, l + m, t + m)).resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    dtype = input_details['dtype']
    scale, zero_point = input_details.get('quantization', (0.0, 0))
    arr_norm = (arr / 127.5) - 1.0
    if scale > 0:
        arr = (arr_norm / scale) + zero_point
    arr = np.clip(arr, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return np.expand_dims(arr, axis=0)


def cpu_worker(duration, counter):
    interpreter = tflite.Interpreter(
        model_path=f"{GPU_TEST_DIR}/mobilenet_v2_qunatized.tflite", num_threads=4
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_data = cpu_preprocess(f"{GPU_TEST_DIR}/test_image.jpg", input_details)
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()  # warm-up

    n = 0
    t_end = time.time() + duration
    while time.time() < t_end:
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details['index'])
        n += 1
    counter['cpu'] = n


# ---------- GPU path: ncnn Vulkan (proven working, float mobilenet) ----------
import ncnn


def gpu_preprocess(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    img = img.crop((l, t, l + m, t + m)).resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


def gpu_worker(duration, counter):
    mat_in = ncnn.Mat(gpu_preprocess(f"{GPU_TEST_DIR}/test_image.jpg")).clone()
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.set_vulkan_device(0)
    net.load_param(f"{GPU_TEST_DIR}/mobilenet_v2.ncnn.param")
    net.load_model(f"{GPU_TEST_DIR}/mobilenet_v2.ncnn.bin")

    ex = net.create_extractor()
    ex.input('in0', mat_in)
    ex.extract('out0')  # warm-up

    n = 0
    t_end = time.time() + duration
    while time.time() < t_end:
        ex = net.create_extractor()
        ex.input('in0', mat_in)
        _, out = ex.extract('out0')
        n += 1
    counter['gpu'] = n


def run_solo(worker, label, key):
    counter = {}
    t0 = time.time()
    worker(DURATION_SEC, counter)
    elapsed = time.time() - t0
    n = counter[key]
    print(f"{label} rieng le: {n} anh trong {elapsed:.2f}s = {n/elapsed:.2f} anh/s ({elapsed/n*1000:.2f} ms/anh)")
    return n / elapsed


def run_concurrent():
    counter = {}
    t_cpu = threading.Thread(target=cpu_worker, args=(DURATION_SEC, counter))
    t_gpu = threading.Thread(target=gpu_worker, args=(DURATION_SEC, counter))
    t0 = time.time()
    t_cpu.start()
    t_gpu.start()
    t_cpu.join()
    t_gpu.join()
    elapsed = time.time() - t0
    n_cpu = counter['cpu']
    n_gpu = counter['gpu']
    total = n_cpu + n_gpu
    print(f"Dong thoi: CPU={n_cpu} anh ({n_cpu/elapsed:.2f} anh/s), GPU={n_gpu} anh ({n_gpu/elapsed:.2f} anh/s), "
          f"TONG={total} anh trong {elapsed:.2f}s = {total/elapsed:.2f} anh/s")
    return n_cpu / elapsed, n_gpu / elapsed, total / elapsed


print("=" * 60)
print("BUOC 1: Do rieng le")
print("=" * 60)
cpu_solo_rate = run_solo(cpu_worker, "CPU", 'cpu')
gpu_solo_rate = run_solo(gpu_worker, "GPU", 'gpu')

print()
print("=" * 60)
print("BUOC 2: Do dong thoi (CPU + GPU song song)")
print("=" * 60)
cpu_conc_rate, gpu_conc_rate, combined_rate = run_concurrent()

print()
print("=" * 60)
print("BANG SO SANH")
print("=" * 60)
print(f"{'':20} | {'anh/giay':>12} | {'ms/anh':>10}")
print("-" * 48)
print(f"{'CPU rieng le':20} | {cpu_solo_rate:12.2f} | {1000/cpu_solo_rate:10.2f}")
print(f"{'GPU rieng le':20} | {gpu_solo_rate:12.2f} | {1000/gpu_solo_rate:10.2f}")
print(f"{'CPU+GPU dong thoi':20} | {combined_rate:12.2f} | {1000/combined_rate:10.2f}")
print("-" * 48)
speedup = combined_rate / cpu_solo_rate
print(f"\nTang toc so voi CPU rieng le: {speedup:.2f}x")
if speedup > 1:
    print(">>> KET HOP CPU+GPU NHANH HON CPU RIENG LE! <<<")
else:
    print(">>> Ket hop KHONG nhanh hon CPU rieng le (co the do tranh chap tai nguyen CPU) <<<")

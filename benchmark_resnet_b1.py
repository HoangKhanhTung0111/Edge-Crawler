import time
import numpy as np
from PIL import Image
import ncnn

N_ITERS = 8


def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


mat_in = ncnn.Mat(preprocess('../gpu_test/test_image.jpg')).clone()

with open('../resnet_test/labels.txt') as f:
    labels = [l.strip() for l in f.readlines()]


def bench(use_vulkan, label):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    if use_vulkan:
        net.set_vulkan_device(0)
    net.load_param('resnet50_b1.ncnn.param')
    net.load_model('resnet50_b1.ncnn.bin')

    ex = net.create_extractor()
    ex.input('in0', mat_in)
    _, out0 = ex.extract('out0')
    out_arr = np.array(out0)
    idx = out_arr.argmax()
    print(f"{label} top1: idx={idx} label={labels[idx] if idx < len(labels) else '?'} score={out_arr[idx]:.2f}")

    t0 = time.time()
    for i in range(N_ITERS):
        ex = net.create_extractor()
        ex.input('in0', mat_in)
        _, out0 = ex.extract('out0')
    total = (time.time() - t0) * 1000
    per_img = total / N_ITERS
    print(f"{label}: {N_ITERS} lan, tong={total:.2f}ms, trung binh={per_img:.2f}ms/anh")
    return per_img


cpu_ms = bench(False, "CPU")
gpu_ms = bench(True, "GPU (Vulkan)")
print(f"\n=== KET QUA: CPU={cpu_ms:.2f}ms  GPU={gpu_ms:.2f}ms  ty le GPU/CPU={gpu_ms/cpu_ms:.2f}x ===")
if gpu_ms < cpu_ms:
    print(">>> GPU NHANH HON CPU! <<<")

import time
import numpy as np
from PIL import Image
import ncnn

BATCH = 4
N_ITERS = 1


def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


single = preprocess('../gpu_test/test_image.jpg')
stacked = np.stack([single] * BATCH, axis=0)
batch_input = np.ascontiguousarray(np.transpose(stacked, (1, 0, 2, 3)))
mat_in = ncnn.Mat(batch_input).clone()

with open('../resnet_test/labels.txt') as f:
    labels = [l.strip() for l in f.readlines()]


def bench(use_vulkan, label):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    if use_vulkan:
        net.set_vulkan_device(0)
    net.load_param('resnet50_b4.ncnn.param')
    net.load_model('resnet50_b4.ncnn.bin')

    t0 = time.time()
    ex = net.create_extractor()
    ex.input('in0', mat_in)
    _, out0 = ex.extract('out0')
    total = (time.time() - t0) * 1000

    out_arr = np.array(out0).reshape(BATCH, -1)
    idx = out_arr[0].argmax()
    print(f"{label} top1 (anh0): idx={idx} label={labels[idx] if idx < len(labels) else '?'} score={out_arr[0][idx]:.2f}")
    per_img = total / BATCH
    print(f"{label}: 1 lo x {BATCH} anh (single call, includes pipeline build), tong={total:.2f}ms, trung binh/anh={per_img:.2f}ms")
    return per_img


cpu_ms = bench(False, "CPU")
gpu_ms = bench(True, "GPU (Vulkan)")
print(f"\n=== KET QUA: CPU={cpu_ms:.2f}ms/anh  GPU={gpu_ms:.2f}ms/anh  ty le GPU/CPU={gpu_ms/cpu_ms:.2f}x ===")
if gpu_ms < cpu_ms:
    print(">>> GPU NHANH HON CPU! <<<")

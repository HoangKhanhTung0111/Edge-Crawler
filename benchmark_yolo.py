import time
import numpy as np
from PIL import Image
import ncnn

N_ITERS = 5


def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((640, 640), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))  # CHW


mat_in = ncnn.Mat(preprocess('../gpu_test/test_image.jpg')).clone()


def bench(use_vulkan, label):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    if use_vulkan:
        net.set_vulkan_device(0)
    net.load_param('yolov11_det.ncnn.param')
    net.load_model('yolov11_det.ncnn.bin')

    ex = net.create_extractor()
    ex.input('in0', mat_in)
    _, out0 = ex.extract('out0')
    boxes = np.array(out0)
    print(f"{label} boxes shape: {boxes.shape}")

    t0 = time.time()
    for i in range(N_ITERS):
        ex = net.create_extractor()
        ex.input('in0', mat_in)
        _, out0 = ex.extract('out0')
    total = (time.time() - t0) * 1000
    print(f"{label}: {N_ITERS} lan, tong={total:.2f} ms, trung binh={total/N_ITERS:.2f} ms/lan")

    boxes = np.array(out0)
    print(f"{label} sample box[0]: {boxes[0, 0] if boxes.ndim == 2 else boxes.reshape(-1,4)[0]}")
    return total / N_ITERS


cpu_ms = bench(False, "CPU")
gpu_ms = bench(True, "GPU (Vulkan)")
print(f"\n=== KET QUA: CPU={cpu_ms:.2f}ms  GPU={gpu_ms:.2f}ms  ty le={gpu_ms/cpu_ms:.2f}x ===")

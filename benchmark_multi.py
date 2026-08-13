import time
import numpy as np
from PIL import Image
import ncnn

N = 20


def preprocess(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    img = img.crop((l, t, l + m, t + m)).resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


mat_in = ncnn.Mat(preprocess('test_image.jpg')).clone()


def bench(use_vulkan, label):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    if use_vulkan:
        net.set_vulkan_device(0)
    net.load_param('mobilenet_v2.ncnn.param')
    net.load_model('mobilenet_v2.ncnn.bin')

    # warm-up
    ex = net.create_extractor()
    ex.input('in0', mat_in)
    ex.extract('out0')

    t0 = time.time()
    for i in range(N):
        ex = net.create_extractor()
        ex.input('in0', mat_in)
        _, out = ex.extract('out0')
    total = (time.time() - t0) * 1000
    print(f"{label}: {N} anh, tong={total:.2f} ms, trung binh={total/N:.2f} ms/anh, throughput={N/(total/1000):.2f} anh/s")


bench(False, "CPU")
bench(True, "GPU (Vulkan)")

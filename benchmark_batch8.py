import time
import numpy as np
from PIL import Image
import ncnn

BATCH = 8
N_ITERS = 5


def preprocess(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    img = img.crop((l, t, l + m, t + m)).resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


single = preprocess('test_image.jpg')
stacked = np.stack([single] * BATCH, axis=0)  # (8,3,224,224) = (N,C,H,W)
batch_input = np.ascontiguousarray(np.transpose(stacked, (1, 0, 2, 3)))  # (3,8,224,224) = (C,N,H,W) for ncnn Mat
print("batch_input shape:", batch_input.shape)


def bench(use_vulkan, label):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    if use_vulkan:
        net.set_vulkan_device(0)
    net.load_param('mobilenet_v2_b8.ncnn.param')
    net.load_model('mobilenet_v2_b8.ncnn.bin')

    mat_in = ncnn.Mat(batch_input).clone()

    # warm-up
    ex = net.create_extractor()
    ex.input('in0', mat_in)
    _, out0 = ex.extract('out0')
    out_arr = np.array(out0)
    print(f"{label} output shape: {out_arr.shape}")

    t0 = time.time()
    for i in range(N_ITERS):
        ex = net.create_extractor()
        ex.input('in0', mat_in)
        _, out0 = ex.extract('out0')
    total = (time.time() - t0) * 1000
    per_batch = total / N_ITERS
    per_image = per_batch / BATCH
    print(f"{label}: {N_ITERS} lo x {BATCH} anh, tong={total:.2f} ms, "
          f"trung binh/lo={per_batch:.2f} ms, trung binh/anh={per_image:.2f} ms, "
          f"throughput={BATCH*N_ITERS/(total/1000):.2f} anh/s")

    out_arr = np.array(out0)
    top1 = np.argmax(out_arr.reshape(BATCH, -1)[0])
    print(f"{label} sanity top1 class idx (anh dau tien trong lo): {top1}")


bench(False, "CPU (batch=8)")
bench(True, "GPU Vulkan (batch=8)")

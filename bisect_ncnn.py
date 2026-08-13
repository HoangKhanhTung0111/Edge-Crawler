import os
import numpy as np
from PIL import Image
import ncnn

BASE_DIR = "/home/arduino/gpu_test"
PARAM_PATH = os.path.join(BASE_DIR, 'mobilenet_v2.ncnn.param')
BIN_PATH = os.path.join(BASE_DIR, 'mobilenet_v2.ncnn.bin')
IMAGE_PATH = os.path.join(BASE_DIR, 'test_image.jpg')


def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
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
    arr = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
    return arr


def extract_blob(use_vulkan, blob_name):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    if use_vulkan:
        net.set_vulkan_device(0)
    net.load_param(PARAM_PATH)
    net.load_model(BIN_PATH)

    img_chw = load_and_preprocess_image(IMAGE_PATH)
    mat_in = ncnn.Mat(img_chw).clone()

    ex = net.create_extractor()
    ex.input("in0", mat_in)
    _, out = ex.extract(blob_name)
    arr = np.array(out)
    del ex
    del net
    return arr


BLOBS_TO_CHECK = ["4", "10", "16", "25", "40", "52", "61", "76", "86", "87", "out0"]

for blob in BLOBS_TO_CHECK:
    try:
        cpu_out = extract_blob(False, blob)
        gpu_out = extract_blob(True, blob)
    except Exception as e:
        print(f"blob {blob}: ERROR {e}")
        continue
    if cpu_out.shape != gpu_out.shape:
        print(f"blob {blob}: SHAPE MISMATCH cpu={cpu_out.shape} gpu={gpu_out.shape}")
        continue
    diff = np.abs(cpu_out.astype(np.float32) - gpu_out.astype(np.float32))
    rel = diff.max() / (np.abs(cpu_out).max() + 1e-8)
    print(f"blob {blob}: shape={cpu_out.shape} cpu[min/max/mean]=({cpu_out.min():.4f},{cpu_out.max():.4f},{cpu_out.mean():.4f}) "
          f"gpu[min/max/mean]=({gpu_out.min():.4f},{gpu_out.max():.4f},{gpu_out.mean():.4f}) max_abs_diff={diff.max():.6f} rel={rel:.4f}")

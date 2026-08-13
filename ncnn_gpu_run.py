import os
import time
import numpy as np
from PIL import Image
import ncnn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_PATH = os.path.join(BASE_DIR, 'mobilenet_v2.ncnn.param')
BIN_PATH = os.path.join(BASE_DIR, 'mobilenet_v2.ncnn.bin')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')
IMAGE_PATH = os.path.join(BASE_DIR, 'test_image.jpg')


def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


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
    arr = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))  # HWC -> CHW
    return arr


def main():
    labels = load_labels(LABELS_PATH)
    print("-> GPU count:", ncnn.get_gpu_count())

    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.set_vulkan_device(0)

    net.load_param(PARAM_PATH)
    net.load_model(BIN_PATH)

    img_chw = load_and_preprocess_image(IMAGE_PATH)
    mat_in = ncnn.Mat(img_chw).clone()

    ex = net.create_extractor()
    ex.input("in0", mat_in)

    print("-> Bat dau Inference (warm-up)...")
    _, out0 = ex.extract("out0")

    ex2 = net.create_extractor()
    ex2.input("in0", mat_in)
    start_time = time.time()
    _, out0 = ex2.extract("out0")
    inference_time = (time.time() - start_time) * 1000
    print(f"-> Hoan thanh trong: {inference_time:.2f} ms")

    output_data = np.array(out0).squeeze()
    print("-> raw output min/max/mean:", output_data.min(), output_data.max(), output_data.mean())
    print("-> has NaN:", np.isnan(output_data).any(), "has Inf:", np.isinf(output_data).any())

    exp_scores = np.exp(output_data - np.max(output_data))
    probabilities = exp_scores / np.sum(exp_scores)
    top_5_indices = np.argsort(probabilities)[-5:][::-1]

    print("=" * 45)
    print("KET QUA PHAN LOAI (TOP 5):")
    for i, idx in enumerate(top_5_indices):
        class_name = labels[idx] if (labels and idx < len(labels)) else "Unknown"
        confidence = probabilities[idx] * 100
        print(f"{i+1}. {class_name} (Index: {idx}) - {confidence:.2f}%")
    print("=" * 45)


if __name__ == '__main__':
    main()

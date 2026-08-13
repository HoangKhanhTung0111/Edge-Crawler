import sys
import time
import numpy as np
from PIL import Image
import ai_edge_litert.interpreter as tflite

MODEL_PATH = "wire_classifier_quantized.tflite"
LABELS = ["lanh (intact)", "dut (broken)"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path, input_details):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    img = img.crop((l, t, l + m, t + m)).resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW (model expects NCHW, exported directly from PyTorch)
    arr = np.expand_dims(arr, 0)

    dtype = input_details['dtype']
    if dtype == np.float32:
        return arr.astype(np.float32)
    scale, zero_point = input_details.get('quantization', (0.0, 0))
    q = arr / scale + zero_point
    info = np.iinfo(dtype)
    q = np.clip(q, info.min, info.max).astype(dtype)
    return q


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_wire.jpg"

    interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    data = preprocess(image_path, inp)
    interp.set_tensor(inp['index'], data)
    interp.invoke()  # warm-up

    N = 20
    t0 = time.time()
    for _ in range(N):
        interp.set_tensor(inp['index'], data)
        interp.invoke()
        output_data = interp.get_tensor(out['index'])
    dt = (time.time() - t0) * 1000 / N

    if out['dtype'] != np.float32:
        oscale, ozp = out.get('quantization', (0.0, 0))
        output_data = (output_data.astype(np.float32) - ozp) * oscale
    logits = np.squeeze(output_data)
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    pred = int(probs.argmax())

    print(f"Anh: {image_path}")
    print(f"Ket qua: {LABELS[pred]}  (lanh={probs[0]*100:.1f}%, dut={probs[1]*100:.1f}%)")
    print(f"Inference time: {dt:.2f} ms/anh")


if __name__ == '__main__':
    main()

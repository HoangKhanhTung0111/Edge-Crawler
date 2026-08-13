import time
import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite

interp = tflite.Interpreter(model_path='mobilenet_v2_qunatized.tflite', num_threads=4)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
dtype = inp['dtype']
scale, zp = inp.get('quantization', (0.0, 0))

with open('labels.txt') as f:
    labels = [l.strip() for l in f.readlines()]


def preprocess_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    m = min(h, w)
    t, l = (h - m) // 2, (w - m) // 2
    img = img[t:t + m, l:l + m]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32)
    arr_norm = (arr / 127.5) - 1.0
    if scale > 0:
        arr = (arr_norm / scale) + zp
    arr = np.clip(arr, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return np.expand_dims(arr, 0)


# warm-up
d = preprocess_cv2('test_image.jpg')
interp.set_tensor(inp['index'], d)
interp.invoke()

N = 30
t0 = time.time()
for i in range(N):
    data = preprocess_cv2('test_image.jpg')
    interp.set_tensor(inp['index'], data)
    interp.invoke()
    output_data = interp.get_tensor(out['index'])
total = (time.time() - t0) * 1000 / N
print(f"Tong pipeline (cv2 preprocess + inference): {total:.2f} ms/anh")

output_data = np.squeeze(output_data)
o_scale, o_zp = out.get('quantization', (0.0, 0))
if o_scale > 0:
    output_data = (output_data.astype(np.float32) - o_zp) * o_scale
exp_scores = np.exp(output_data - np.max(output_data))
probs = exp_scores / np.sum(exp_scores)
idx = probs.argmax()
print(f"Top1: {labels[idx]} ({probs[idx]*100:.2f}%)")

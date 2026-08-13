import os
import time
import numpy as np
from PIL import Image
import pyarmnn as ann

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mobilenet_v2.tflite')
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
    return np.expand_dims(arr, axis=0)


def main():
    labels = load_labels(LABELS_PATH)

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(MODEL_PATH)

    graph_id = 0
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    print("-> Input tensors:", list(input_names))
    print("-> Output tensors:", list(output_names))

    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    output_binding_info = parser.GetNetworkOutputBindingInfo(graph_id, output_names[0])

    preferred_backends = [ann.BackendId('GpuAcc'), ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    opt_network, messages = ann.Optimize(
        network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions()
    )
    for m in messages:
        print("-> Optimize message:", m)

    net_id, _ = runtime.LoadNetwork(opt_network)
    print(f"-> Da nap network, id={net_id}")

    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    print("-> Backend duoc gan cho input:", runtime.GetDeviceSpec())

    img_array = load_and_preprocess_image(IMAGE_PATH)
    input_tensors = ann.make_input_tensors([input_binding_info], [img_array])
    output_tensors = ann.make_output_tensors([output_binding_info])

    print("-> Bat dau Inference (warm-up)...")
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)

    start_time = time.time()
    runtime.EnqueueWorkload(net_id, input_tensors, output_tensors)
    inference_time = (time.time() - start_time) * 1000
    print(f"-> Hoan thanh trong: {inference_time:.2f} ms")

    output_data = ann.workload_tensors_to_ndarray(output_tensors)[0]
    output_data = np.squeeze(output_data)

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

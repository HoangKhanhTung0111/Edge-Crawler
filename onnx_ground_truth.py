import onnxruntime as ort
import numpy as np
from PIL import Image

MODEL_PATH = "mobilenet_v2.onnx"
IMAGE_PATH = "test_image.jpg"
LABELS_PATH = "labels.txt"


def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
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
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def main():
    with open(LABELS_PATH) as f:
        labels = [l.strip() for l in f.readlines()]

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_data = preprocess(IMAGE_PATH)

    outputs = session.run(None, {input_name: input_data})
    output_data = np.squeeze(outputs[0])
    print("output shape:", output_data.shape)

    exp_scores = np.exp(output_data - np.max(output_data))
    probabilities = exp_scores / np.sum(exp_scores)
    top_5 = np.argsort(probabilities)[-5:][::-1]

    for i, idx in enumerate(top_5):
        name = labels[idx] if idx < len(labels) else "Unknown"
        print(f"{i+1}. {name} (Index: {idx}) - {probabilities[idx]*100:.2f}%")


if __name__ == '__main__':
    main()

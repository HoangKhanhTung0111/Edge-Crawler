import numpy as np
from PIL import Image
import ai_edge_litert.interpreter as tflite
import time
import os
import ctypes
import ctypes.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mobilenet_v2_qunatized.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')
IMAGE_PATH = os.path.join(BASE_DIR, 'test_image.jpg')
GPU_DELEGATE_PATH = os.path.join(BASE_DIR, 'libtensorflowlite_gpu_delegate.so')


def setup_gpu_environment():
    os.environ["RUSTICL_ENABLE"] = "adreno,freedreno,fd"
    os.environ["OCL_ICD_VENDORS"] = "/etc/OpenCL/vendors"
    try:
        ctypes.CDLL("libGLESv2.so.2", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libEGL.so.1", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass
    cl_lib = ctypes.util.find_library("OpenCL") or "libOpenCL.so.1"
    try:
        ctypes.CDLL(cl_lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


class TfLiteGpuDelegateOptionsV2(ctypes.Structure):
    _fields_ = [
        ("is_precision_loss_allowed", ctypes.c_int32),
        ("inference_preference", ctypes.c_int32),
        ("inference_priority1", ctypes.c_int32),
        ("inference_priority2", ctypes.c_int32),
        ("inference_priority3", ctypes.c_int32),
        ("experimental_flags", ctypes.c_int64),
        ("max_delegated_partitions", ctypes.c_int32),
        ("serialization_dir", ctypes.c_char_p),
        ("model_token", ctypes.c_char_p),
    ]


class CustomGpuDelegate:
    def __init__(self, library_path):
        self._library = ctypes.CDLL(library_path)
        self._library.TfLiteGpuDelegateV2Create.argtypes = [ctypes.POINTER(TfLiteGpuDelegateOptionsV2)]
        self._library.TfLiteGpuDelegateV2Create.restype = ctypes.c_void_p
        self._library.TfLiteGpuDelegateV2Delete.argtypes = [ctypes.c_void_p]

        options = TfLiteGpuDelegateOptionsV2()
        options.is_precision_loss_allowed = 1
        options.inference_preference = 0
        options.inference_priority1 = 2
        options.inference_priority2 = 1
        options.inference_priority3 = 0
        options.experimental_flags = 3  # CL_ONLY | ENABLE_QUANT
        options.max_delegated_partitions = 1
        options.serialization_dir = None
        options.model_token = None

        self._delegate_ptr = self._library.TfLiteGpuDelegateV2Create(ctypes.byref(options))
        if not self._delegate_ptr:
            raise RuntimeError("Khoi tao GPU Delegate that bai.")

    def _get_native_delegate_pointer(self):
        return self._delegate_ptr

    def __del__(self):
        if hasattr(self, '_delegate_ptr') and self._delegate_ptr and hasattr(self, '_library'):
            self._library.TfLiteGpuDelegateV2Delete(self._delegate_ptr)


def load_labels(label_path):
    try:
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception:
        return []


def load_and_preprocess_image(image_path, input_details):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)

    dtype = input_details['dtype']
    if dtype == np.float32:
        img_array = (img_array / 127.5) - 1.0
    else:
        scale, zero_point = input_details.get('quantization', (0.0, 0))
        img_normalized = (img_array / 127.5) - 1.0
        if scale > 0:
            img_array = (img_normalized / scale) + zero_point
        img_array = np.clip(img_array, np.iinfo(dtype).min, np.iinfo(dtype).max)
        img_array = img_array.astype(dtype)

    return np.expand_dims(img_array, axis=0)


def main():
    setup_gpu_environment()
    labels = load_labels(LABELS_PATH)

    print("-> Khoi tao LiteRT Interpreter voi GPU delegate...")
    delegates = []
    if os.path.exists(GPU_DELEGATE_PATH):
        try:
            gpu_delegate = CustomGpuDelegate(GPU_DELEGATE_PATH)
            delegates.append(gpu_delegate)
            print("-> Nap GPU Delegate thanh cong!")
        except Exception as e:
            print(f"-> LOI PHAN CUNG: {e}")

    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=delegates)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Loi khi load model: {e}")
        return

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_data = load_and_preprocess_image(IMAGE_PATH, input_details)
    interpreter.set_tensor(input_details['index'], input_data)

    print("-> Bat dau Inference (warm-up)...")
    interpreter.invoke()

    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000
    print(f"-> Hoan thanh trong: {inference_time:.2f} ms")

    output_data = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] != np.float32:
        scale, zero_point = output_details.get('quantization', (0.0, 0))
        if scale > 0:
            output_data = (output_data.astype(np.float32) - zero_point) * scale

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

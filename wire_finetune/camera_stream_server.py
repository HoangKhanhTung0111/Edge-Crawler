"""
Stream camera feed over HTTP (MJPEG) with wire-classifier prediction overlaid
top-left. Access from any browser on the same network at http://<device-ip>:PORT/

Architecture: ONE dedicated thread reads the camera continuously and publishes
the latest frame. Inference and streaming both consume that shared frame
instead of calling cap.read() themselves - cv2.VideoCapture is not safe to
read from multiple threads at once, which was causing the stutter.
"""
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response
import ai_edge_litert.interpreter as tflite

MODEL_PATH = "/home/arduino/wire_classifier/wire_classifier_quantized.tflite"
CAMERA_INDEX = 2  # Logitech HD Webcam C615 -> /dev/video2
PORT = 8080
LABELS = ["LANH (intact)", "DUT (broken)"]
LABEL_COLORS = [(0, 200, 0), (0, 0, 255)]  # BGR: green for lanh, red for dut

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = Flask(__name__)

interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interp.allocate_tensors()
inp_details = interp.get_input_details()[0]
out_details = interp.get_output_details()[0]
INPUT_DTYPE = inp_details['dtype']

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"Khong mo duoc camera index {CAMERA_INDEX}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the newest frame, not a queued old one

# --- shared state: only capture_loop() ever touches `cap` ---
frame_lock = threading.Lock()
latest_frame = {"frame": None, "capture_fps": 0.0}

state_lock = threading.Lock()
state = {"label": "...", "conf": 0.0, "color": (200, 200, 200), "ms": 0.0, "infer_fps": 0.0}


def capture_loop():
    n = 0
    t_win = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame["frame"] = frame
        n += 1
        now = time.time()
        if now - t_win >= 1.0:
            with frame_lock:
                latest_frame["capture_fps"] = n / (now - t_win)
            n = 0
            t_win = now


def get_latest_frame():
    with frame_lock:
        f = latest_frame["frame"]
        return None if f is None else f.copy()


def preprocess(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    m = min(h, w)
    t, l = (h - m) // 2, (w - m) // 2
    img = img[t:t + m, l:l + m]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW (model expects NCHW)
    arr = np.expand_dims(arr, 0)

    if INPUT_DTYPE == np.float32:
        return arr.astype(np.float32)
    scale, zero_point = inp_details.get('quantization', (0.0, 0))
    q = arr / scale + zero_point
    info = np.iinfo(INPUT_DTYPE)
    return np.clip(q, info.min, info.max).astype(INPUT_DTYPE)


def classify(frame_bgr):
    data = preprocess(frame_bgr)
    t0 = time.time()
    interp.set_tensor(inp_details['index'], data)
    interp.invoke()
    output = interp.get_tensor(out_details['index'])
    dt = (time.time() - t0) * 1000

    if out_details['dtype'] != np.float32:
        oscale, ozp = out_details.get('quantization', (0.0, 0))
        output = (output.astype(np.float32) - ozp) * oscale
    logits = np.squeeze(output)
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    pred = int(probs.argmax())
    return LABELS[pred], float(probs[pred]), LABEL_COLORS[pred], dt


def inference_loop():
    n = 0
    t_win = time.time()
    while True:
        frame = get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        label, conf, color, ms = classify(frame)
        n += 1
        now = time.time()
        infer_fps = None
        if now - t_win >= 1.0:
            infer_fps = n / (now - t_win)
            n = 0
            t_win = now
        with state_lock:
            state["label"] = label
            state["conf"] = conf
            state["color"] = color
            state["ms"] = ms
            if infer_fps is not None:
                state["infer_fps"] = infer_fps


def mjpeg_generator():
    while True:
        frame = get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        with state_lock:
            label, conf, color, ms, infer_fps = (
                state["label"], state["conf"], state["color"], state["ms"], state["infer_fps"])
        with frame_lock:
            cap_fps = latest_frame["capture_fps"]

        text = f"{label} {conf*100:.1f}%"
        cv2.rectangle(frame, (5, 5), (10 + 12 * len(text), 65), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{ms:.0f} ms  |  cam {cap_fps:.1f} fps  |  infer {infer_fps:.1f} fps",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return (
        "<html><head><title>Wire Classifier Live</title></head>"
        "<body style='background:#111;text-align:center;'>"
        "<h2 style='color:#eee;font-family:sans-serif'>UnoQ Wire Classifier</h2>"
        "<img src='/stream' style='max-width:95%;border:2px solid #444'>"
        "</body></html>"
    )


@app.route('/stream')
def stream():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    print(f"Server dang chay tai http://0.0.0.0:{PORT}  (truy cap qua http://172.16.3.88:{PORT})")
    app.run(host='0.0.0.0', port=PORT, threaded=True)

"""
Wire-defect inspection robot.

Behaviour (as requested):
  - drives forward continuously
  - intact wire -> ignored, keeps driving
  - broken wire, seen for a few consecutive frames -> stop, pivot to hunt for
    the pivot angle that maximises the classifier's "broken" confidence
    (best available proxy for "wire centered in frame" since the model is a
    classifier, not a detector with bounding boxes), photograph, log, undo
    the pivot to restore heading, resume driving.

Also serves an MJPEG monitor with the live detection + robot state overlay
on port 8080, reusing the single-capture-thread pattern from
camera_stream_server.py (cv2.VideoCapture must only ever be read from one
thread).
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _ensure(pkg_import_name, pip_name):
    if importlib.util.find_spec(pkg_import_name) is None:
        uv_path = shutil.which("uv")
        if uv_path is None:
            raise RuntimeError("App Lab khong tim thay uv de cai dat dependency")
        print(f"[startup] Installing {pip_name}...")
        subprocess.check_call([uv_path, "pip", "install", "--python", sys.executable, pip_name])


_ensure("flask", "Flask==3.1.3")
_ensure("ai_edge_litert", "ai-edge-litert==2.2.0")

import cv2
import numpy as np
from flask import Flask, Response
import ai_edge_litert.interpreter as tflite

from arduino.app_utils import App, Bridge

# ---------------------------------------------------------------- config ---
# App runs inside a container that only bind-mounts its own app folder, so
# paths must live under here (not the host's /home/arduino/wire_classifier).
APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(APP_ROOT / "python" / "models" / "wire_classifier_quantized.tflite")
# /dev/videoN indices for the USB webcam shift across reboots (the SoC's own
# hardware video codec devices share the same /dev/video* numbering), so we
# open it via the udev by-id symlink instead, which is stable per USB device.
CAMERA_BY_ID_GLOB = "usb-046d_HD_Webcam_C615_*-video-index0"
PORT = 8080
LOG_DIR = APP_ROOT / "inspection_log"

LABELS = ["INTACT", "BROKEN"]
LABEL_COLORS = [(0, 200, 0), (0, 0, 255)]  # BGR: green for intact, red for broken

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Motion modes (must match sketch/sketch.ino's Motion enum)
MOTION_STOP = 0
MOTION_FORWARD = 1
MOTION_PIVOT_RIGHT = 2
MOTION_PIVOT_LEFT = 3

BROKEN_CONF_THRESHOLD = 0.80  # min confidence to count a frame as "broken"
CONSEC_REQUIRED = 3           # consecutive qualifying frames before triggering a stop
COOLDOWN_SEC = 4.0            # ignore new triggers for this long after resuming

SCAN_STEP_MS = 100            # pivot burst duration per centering step
SCAN_SETTLE_S = 0.25          # wait after each pivot for a fresh classified frame
SCAN_MAX_STEPS = 8
SCAN_TARGET_CONF = 0.97

# Set True to run camera + sensors + full decision logic with the motors
# kept completely silent - useful for bench-testing detection without the
# car actually driving off.
TESTING_MOTORS_DISABLED = False

# Single forward-facing sensor - no way to tell which side has more room, so
# obstacles are always avoided by turning the same fixed direction.
OBSTACLE_THRESHOLD_CM = 15.0   # trigger avoidance when something is closer than this
OBSTACLE_CONSEC_REQUIRED = 2  # consecutive close readings before triggering (debounce)
AVOID_DIRECTION = MOTION_PIVOT_RIGHT
AVOID_TURN_MS = 400           # pivot burst duration per avoidance step
AVOID_MAX_STEPS = 6           # give up turning after this many steps (don't spin forever)

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ AI + camera ---
interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interp.allocate_tensors()
inp_details = interp.get_input_details()[0]
out_details = interp.get_output_details()[0]
INPUT_DTYPE = inp_details['dtype']

def open_camera():
    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.is_dir():
        for entry in sorted(by_id_dir.glob(CAMERA_BY_ID_GLOB)):
            c = cv2.VideoCapture(str(entry), cv2.CAP_V4L2)
            if c.isOpened():
                return c
            c.release()
    # fallback: scan raw indices for anything that opens and actually reads a frame
    for idx in range(10):
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            ok, _ = c.read()
            if ok:
                return c
        c.release()
    return None


cap = open_camera()
if cap is None:
    raise RuntimeError("Khong tim thay camera nao kha dung (by-id lan fallback index deu that bai)")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_lock = threading.Lock()
latest_frame = {"frame": None, "capture_fps": 0.0}

state_lock = threading.Lock()
state = {"label": "...", "conf": 0.0, "color": (200, 200, 200), "ms": 0.0, "infer_fps": 0.0}

robot_state = {"mode": "BOOTING"}


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


interp_lock = threading.Lock()


def classify(frame_bgr):
    data = preprocess(frame_bgr)
    t0 = time.time()
    with interp_lock:
        interp.set_tensor(inp_details['index'], data)
        interp.invoke()
        output = interp.get_tensor(out_details['index']).copy()
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


def classify_latest(min_wait=0.0):
    if min_wait > 0:
        time.sleep(min_wait)
    frame = get_latest_frame()
    if frame is None:
        return "...", 0.0
    label, conf, _, _ = classify(frame)
    return label, conf


# ------------------------------------------------------------ robot logic ---
def send_motion(mode):
    if TESTING_MOTORS_DISABLED:
        return
    Bridge.call("set_motion", mode)


def pivot_burst(mode, ms):
    send_motion(mode)
    time.sleep(ms / 1000.0)
    send_motion(MOTION_STOP)


def center_on_defect(initial_conf):
    """Hill-climb the pivot angle that maximises 'broken' confidence, as a
    proxy for having the broken wire centered in frame (the model is a
    classifier without bounding boxes, so this is the best-effort approach).
    Returns (net_pivot_ms signed +right/-left, best_conf)."""
    direction = MOTION_PIVOT_RIGHT
    best_conf = initial_conf
    net_pivot_ms = 0
    reversed_once = False

    for _ in range(SCAN_MAX_STEPS):
        sign = 1 if direction == MOTION_PIVOT_RIGHT else -1
        pivot_burst(direction, SCAN_STEP_MS)
        net_pivot_ms += sign * SCAN_STEP_MS

        label, conf = classify_latest(SCAN_SETTLE_S)
        improved = label.startswith("BROKEN") and conf > best_conf

        if improved:
            best_conf = conf
        elif not reversed_once:
            direction = MOTION_PIVOT_LEFT if direction == MOTION_PIVOT_RIGHT else MOTION_PIVOT_RIGHT
            reversed_once = True
        else:
            break

        if best_conf >= SCAN_TARGET_CONF:
            break

    return net_pivot_ms, best_conf


def inspect_broken_wire(trigger_conf):
    robot_state["mode"] = "STOPPED - INSPECTING"
    send_motion(MOTION_STOP)
    time.sleep(0.2)

    net_pivot_ms, best_conf = center_on_defect(trigger_conf)

    frame = get_latest_frame()
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"broken_{ts}_conf{int(best_conf * 100)}.jpg"
    if frame is not None:
        cv2.imwrite(str(LOG_DIR / fname), frame)

    entry = {
        "timestamp": ts,
        "confidence": round(best_conf, 4),
        "image": fname,
        "pivot_ms": net_pivot_ms,
    }
    with open(LOG_DIR / "inspections.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[inspect] saved {fname} conf={best_conf:.2f} pivot_ms={net_pivot_ms}")

    # undo the pivot so the car resumes roughly its original heading
    undo_ms = abs(net_pivot_ms)
    if undo_ms > 0:
        undo_dir = MOTION_PIVOT_LEFT if net_pivot_ms > 0 else MOTION_PIVOT_RIGHT
        pivot_burst(undo_dir, undo_ms)

    robot_state["mode"] = "DRIVING"
    send_motion(MOTION_FORWARD)


def get_distance_cm():
    try:
        return Bridge.call("get_distance_cm")
    except Exception as error:
        print(f"[distance] {type(error).__name__}: {error}")
        return -1.0


def avoid_obstacle(trigger_dist):
    """Something is too close ahead - stop, turn away (fixed direction, since
    a single forward-facing sensor can't tell which side has more room) until
    the path reads clear again or we give up turning, then resume."""
    print(f"[avoid] obstacle at {trigger_dist:.1f} cm, turning")
    robot_state["mode"] = "AVOIDING OBSTACLE"
    send_motion(MOTION_STOP)
    time.sleep(0.1)

    steps = 0
    dist = trigger_dist
    for steps in range(1, AVOID_MAX_STEPS + 1):
        pivot_burst(AVOID_DIRECTION, AVOID_TURN_MS)
        time.sleep(0.1)
        dist = get_distance_cm()
        if dist < 0 or dist >= OBSTACLE_THRESHOLD_CM:
            break

    print(f"[avoid] done after {steps} step(s), distance now {dist:.1f} cm")
    robot_state["mode"] = "DRIVING"
    send_motion(MOTION_FORWARD)


_consec_broken = 0
_consec_close = 0
_cooldown_until = 0.0
_driving_started = False
_last_heartbeat = 0.0


def control_tick():
    """Called repeatedly by App.run(); one tick of the drive/detect/inspect/
    avoid state machine. Obstacle avoidance takes priority over wire
    inspection since it's about not colliding with something."""
    global _consec_broken, _consec_close, _cooldown_until, _driving_started

    if not _driving_started:
        send_motion(MOTION_FORWARD)
        robot_state["mode"] = "TEST MODE (motors off)" if TESTING_MOTORS_DISABLED else "DRIVING"
        _driving_started = True

    now = time.time()
    if now < _cooldown_until:
        time.sleep(0.1)
        return

    dist = get_distance_cm()
    global _last_heartbeat
    if now - _last_heartbeat >= 1.0:
        print(f"[tick] distance={dist:.1f}cm" if dist >= 0 else "[tick] distance=out-of-range")
        _last_heartbeat = now
    is_close = 0 <= dist < OBSTACLE_THRESHOLD_CM
    _consec_close = _consec_close + 1 if is_close else 0

    if _consec_close >= OBSTACLE_CONSEC_REQUIRED:
        _consec_close = 0
        avoid_obstacle(dist)
        _cooldown_until = time.time() + COOLDOWN_SEC
        return

    with state_lock:
        label, conf = state["label"], state["conf"]

    is_broken = label.startswith("BROKEN") and conf >= BROKEN_CONF_THRESHOLD
    _consec_broken = _consec_broken + 1 if is_broken else 0

    if _consec_broken >= CONSEC_REQUIRED:
        _consec_broken = 0
        inspect_broken_wire(conf)
        _cooldown_until = time.time() + COOLDOWN_SEC

    time.sleep(0.1)


def safe_stop():
    try:
        Bridge.call("set_motion", MOTION_STOP)
    except Exception as error:
        print(f"[safe_stop] {type(error).__name__}: {error}")


# ---------------------------------------------------------------- monitor ---
flask_app = Flask(__name__)


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

        text = f"{label} {conf * 100:.1f}%"
        cv2.rectangle(frame, (5, 5), (10 + 12 * len(text), 90), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{ms:.0f} ms  |  cam {cap_fps:.1f} fps  |  infer {infer_fps:.1f} fps",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, robot_state["mode"], (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@flask_app.route('/')
def index():
    return (
        "<html><head><title>Wire Inspector Robot</title></head>"
        "<body style='background:#111;text-align:center;'>"
        "<h2 style='color:#eee;font-family:sans-serif'>UnoQ Wire Inspector Robot</h2>"
        "<img src='/stream' style='max-width:95%;border:2px solid #444'>"
        "</body></html>"
    )


@flask_app.route('/stream')
def stream():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    flask_app.run(host='0.0.0.0', port=PORT, threaded=True, use_reloader=False)


# --------------------------------------------------------------------------
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()
threading.Thread(target=run_flask, daemon=True).start()

print(f"Monitor: http://172.16.3.88:{PORT}")

try:
    App.run(user_loop=control_tick)
finally:
    safe_stop()

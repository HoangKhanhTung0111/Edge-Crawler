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
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
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
from flask import Flask, Response, abort, send_from_directory
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

BROKEN_CONF_THRESHOLD = 0.85  # min confidence to count a frame as "broken"
# Raised from 0.80 (not all the way to 0.92 that was tried briefly): the
# model was never trained on "no wire in frame at all" (e.g. camera pointed
# at bare floor/carpet after a pivot), so unfamiliar backgrounds sometimes
# get classified BROKEN with real confidence. But training photos were all
# shot close to the wire, so confidence on a genuine broken wire also only
# gets high once the car is already close to it - too high a threshold
# combined with the old wider vote window meant it could still be closing
# in by the time enough high-confidence frames accumulated to react. 0.85
# is a partial filter, not a full fix (see BROKEN_WINDOW below for the
# other half); the real fix is retraining with wire-at-a-distance photos.
#
# A strict "N consecutive" streak was too fragile at driving speed: motion
# blur/vibration would flip a single frame back to INTACT, resetting the
# streak to 0 and letting a genuinely broken wire sail past unconfirmed.
# A sliding-window majority tolerates that kind of one-off flicker - kept
# short (3 frames, need 2) so it also reacts quickly once close enough to
# be confident, rather than waiting through several more frames of driving
# closer.
BROKEN_WINDOW_SIZE = 3        # look at the last N frames
BROKEN_WINDOW_REQUIRED = 2    # ...and trigger once at least this many were "broken"
COOLDOWN_SEC = 4.0            # ignore new triggers for this long after resuming

SCAN_STEP_MS = 100            # pivot burst duration per centering step
SCAN_SETTLE_S = 0.25          # wait after each pivot for a fresh classified frame
SCAN_MAX_STEPS = 8
SCAN_TARGET_CONF = 0.97

# Purely cosmetic "look, it's inspecting thoroughly" wiggle for the demo -
# alternates pivots (net rotation ~0, even step count) after the real photo
# is already saved. No camera capture happens during this - only ever one
# real photo per detection, to keep storage use down.
THEATRICAL_STEPS = 6
THEATRICAL_TURN_MS = 150
THEATRICAL_PAUSE_S = 0.25

# Set True to run camera + sensors + full decision logic with the motors
# kept completely silent - useful for bench-testing detection without the
# car actually driving off.
TESTING_MOTORS_DISABLED = False

# Single forward-facing sensor - no way to tell which side actually has more
# room, so each avoidance picks a random direction rather than always the
# same one (helps avoid getting stuck repeatedly turning into a dead end).
OBSTACLE_THRESHOLD_CM = 15.0   # trigger avoidance when something is closer than this
# Diagnostic logging showed the HC-SR04 reading legitimately fails (times
# out) roughly every other poll, regardless of what's actually in front of
# it - a strict "N consecutive" debounce needs a run of back-to-back good
# reads, which is unlikely if every other one drops, so it could miss a
# real obstacle sitting right in front of it. Sliding-window majority
# (same fix already applied to broken-wire detection) tolerates that.
OBSTACLE_WINDOW_SIZE = 3
OBSTACLE_WINDOW_REQUIRED = 2
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


def configure_camera(c):
    # MJPG is what this webcam natively compresses to on-device; forcing it
    # (fourcc must be set before width/height on some drivers) avoids an
    # extra software format-conversion step some UVC drivers otherwise do,
    # which was showing up as growing latency (stable read-fps, but an
    # increasing lag between a real-world change and it showing up) rather
    # than dropped frames.
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)


_cap0 = open_camera()
if _cap0 is None:
    raise RuntimeError("Khong tim thay camera nao kha dung (by-id lan fallback index deu that bai)")
configure_camera(_cap0)

# Held in a dict (not a bare module-level name) so the watchdog thread can
# swap in a freshly reopened VideoCapture out from under capture_loop().
camera_state = {"cap": _cap0, "last_frame_at": time.time()}
CAMERA_STALL_TIMEOUT_S = 3.0  # no new frame for this long -> assume the driver/USB hung and reopen

frame_lock = threading.Lock()
latest_frame = {"frame": None, "capture_fps": 0.0}

state_lock = threading.Lock()
state = {"label": "...", "conf": 0.0, "color": (200, 200, 200), "ms": 0.0, "infer_fps": 0.0}

robot_state = {"mode": "BOOTING"}


def capture_loop():
    n = 0
    t_win = time.time()
    last_signature = None
    while True:
        # Reverted the extra-grab() "drain" attempt: grab() blocks waiting
        # for the next frame regardless of whether the wait is because of a
        # real backlog or just because no new frame exists yet - there's no
        # portable way to tell those apart, so each extra grab() was really
        # just an extra wait for a whole new frame period. 3 extra grabs
        # every iteration meant waiting for 4 frame-arrivals per frame
        # actually processed, quartering the effective fps (15 -> ~3-4,
        # matching exactly what was reported).
        ok, frame = camera_state["cap"].read()
        if not ok:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame["frame"] = frame
        # A "successful" read() doesn't guarantee a genuinely NEW frame - if
        # the driver ever gets stuck handing back the same buffered image
        # forever, reads keep succeeding (so a read-failure-only watchdog
        # never fires) while the stream visibly freezes. Only count this as
        # a live frame if the content actually changed (cheap downsampled
        # signature, not a full-frame compare) - real camera sensor noise
        # means even a static scene never repeats byte-for-byte, so this is
        # safe against false "stalled" triggers.
        signature = frame[::20, ::20].tobytes()
        if signature != last_signature:
            camera_state["last_frame_at"] = time.time()
            last_signature = signature
        n += 1
        now = time.time()
        if now - t_win >= 1.0:
            with frame_lock:
                latest_frame["capture_fps"] = n / (now - t_win)
            n = 0
            t_win = now


def camera_watchdog_loop():
    """cap.read() can hang indefinitely on a USB/V4L2 glitch without ever
    raising - no exception, no timeout, the stream just visibly freezes on
    the last good frame forever. Detect that (no new frame for a while) and
    force a reopen. Swap in the new capture object *before* releasing the
    old one - releasing a VideoCapture from another thread while it's stuck
    inside a blocking read() is what actually unblocks that stuck call."""
    while True:
        time.sleep(1.0)
        if time.time() - camera_state["last_frame_at"] < CAMERA_STALL_TIMEOUT_S:
            continue
        print("[camera] stalled, reopening...")
        old_cap = camera_state["cap"]
        new_cap = open_camera()
        if new_cap is None:
            print("[camera] reopen failed, will retry")
            camera_state["last_frame_at"] = time.time()  # avoid a tight retry loop
            continue
        configure_camera(new_cap)
        camera_state["cap"] = new_cap
        camera_state["last_frame_at"] = time.time()
        try:
            old_cap.release()
        except Exception as error:
            print(f"[camera] error releasing stalled capture: {type(error).__name__}: {error}")
        print("[camera] reopened")


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


def draw_label(frame, label, conf, color):
    """Burns the classification into the top-left corner. Used for both the
    live MJPEG overlay and the photo saved to inspection_log/, so a saved
    photo shows the same "BROKEN 94%"-style readout an evidence photo needs,
    not a bare frame."""
    text = f"{label} {conf * 100:.1f}%"
    cv2.rectangle(frame, (5, 5), (10 + 12 * len(text), 42), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame


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
        draw_label(frame, "BROKEN", best_conf, LABEL_COLORS[1])
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

    # cosmetic only - the real photo is already saved above
    for i in range(THEATRICAL_STEPS):
        d = MOTION_PIVOT_LEFT if i % 2 == 0 else MOTION_PIVOT_RIGHT
        pivot_burst(d, THEATRICAL_TURN_MS)
        time.sleep(THEATRICAL_PAUSE_S)

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
    """Something is too close ahead - stop, turn away (random direction each
    time, since a single forward-facing sensor can't tell which side has
    more room) until the path reads clear again or we give up turning, then
    resume."""
    direction = random.choice((MOTION_PIVOT_LEFT, MOTION_PIVOT_RIGHT))
    print(f"[avoid] obstacle at {trigger_dist:.1f} cm, turning "
          f"{'left' if direction == MOTION_PIVOT_LEFT else 'right'}")
    robot_state["mode"] = "AVOIDING OBSTACLE"
    send_motion(MOTION_STOP)
    time.sleep(0.1)

    steps = 0
    dist = trigger_dist
    for steps in range(1, AVOID_MAX_STEPS + 1):
        pivot_burst(direction, AVOID_TURN_MS)
        time.sleep(0.1)
        dist = get_distance_cm()
        if dist < 0 or dist >= OBSTACLE_THRESHOLD_CM:
            break

    print(f"[avoid] done after {steps} step(s), distance now {dist:.1f} cm")
    robot_state["mode"] = "DRIVING"
    send_motion(MOTION_FORWARD)


_broken_window = deque(maxlen=BROKEN_WINDOW_SIZE)
_close_window = deque(maxlen=OBSTACLE_WINDOW_SIZE)
_cooldown_until = 0.0
_driving_started = False
_last_heartbeat = 0.0


def control_tick():
    """Called repeatedly by App.run(); one tick of the drive/detect/inspect/
    avoid state machine. Obstacle avoidance takes priority over wire
    inspection since it's about not colliding with something."""
    global _cooldown_until, _driving_started

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
    _close_window.append(is_close)

    if sum(_close_window) >= OBSTACLE_WINDOW_REQUIRED:
        _close_window.clear()
        avoid_obstacle(dist)
        _cooldown_until = time.time() + COOLDOWN_SEC
        return

    with state_lock:
        label, conf = state["label"], state["conf"]

    is_broken = label.startswith("BROKEN") and conf >= BROKEN_CONF_THRESHOLD
    _broken_window.append(is_broken)

    if sum(_broken_window) >= BROKEN_WINDOW_REQUIRED:
        _broken_window.clear()
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

        cv2.rectangle(frame, (5, 46), (330, 90), (0, 0, 0), -1)
        draw_label(frame, label, conf, color)
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
        "<p><a href='/gallery' style='color:#6cf;font-family:sans-serif'>View captured detections &rarr;</a></p>"
        "</body></html>"
    )


@flask_app.route('/gallery')
def gallery():
    entries = []
    log_path = LOG_DIR / "inspections.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
    entries.reverse()  # most recent first

    cards = "".join(
        f"<div style='display:inline-block;margin:10px;text-align:center'>"
        f"<img src='/photos/{e['image']}' style='max-width:320px;border:2px solid #444;border-radius:4px'><br>"
        f"<span style='color:#ccc;font-family:sans-serif;font-size:14px'>"
        f"{e['timestamp']} &middot; {e['confidence'] * 100:.1f}%</span></div>"
        for e in entries
    )
    if not cards:
        cards = "<p style='color:#888;font-family:sans-serif'>No broken-wire detections logged yet.</p>"

    return (
        "<html><head><title>Detections - Wire Inspector Robot</title></head>"
        "<body style='background:#111;text-align:center'>"
        "<h2 style='color:#eee;font-family:sans-serif'>Broken-wire detections "
        f"({len(entries)})</h2>"
        "<p><a href='/' style='color:#6cf;font-family:sans-serif'>&larr; back to live stream</a></p>"
        f"{cards}"
        "</body></html>"
    )


@flask_app.route('/photos/<path:filename>')
def photo(filename):
    if "/" in filename or "\\" in filename:
        abort(404)
    return send_from_directory(str(LOG_DIR), filename)


@flask_app.route('/stream')
def stream():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    flask_app.run(host='0.0.0.0', port=PORT, threaded=True, use_reloader=False)


# --------------------------------------------------------------------------
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=camera_watchdog_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()
threading.Thread(target=run_flask, daemon=True).start()

print(f"Monitor: http://172.16.3.88:{PORT}")

try:
    App.run(user_loop=control_tick)
finally:
    safe_stop()

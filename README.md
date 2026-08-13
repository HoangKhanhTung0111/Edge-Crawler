# Wire Defect Inspection Robot (Arduino UNO Q)

An autonomous robot that drives forward on its own, uses an on-device AI
model to tell intact wire from broken wire through its camera, and reacts:
ignore intact wire and keep going, or stop, turn to face the broken wire,
photograph it, log it, and resume.

Built on the [Arduino UNO Q](https://docs.arduino.cc/hardware/uno-q/)
(Qualcomm Dragonwing QRB2210 running Linux + an STM32U585 microcontroller
running Zephyr/Arduino sketches, bridged by the App Lab RPC framework).

## Repo layout

```
wire_finetune/    training pipeline: raw photos -> finetuned, quantized model
robot_app/        the deployed Arduino App Lab app that runs on the robot
```

The `benchmarks` branch of this repo holds an earlier, separate line of work:
CPU/GPU/NPU inference-time benchmarking scripts across several architectures
(MobileNet, ResNet, EfficientNet, YOLO). It's kept apart from `main` because
it's exploratory measurement code, not part of the robot itself.

## How it works

1. **Model**: MobileNetV2 with a frozen ImageNet backbone and a retrained
   2-class classifier head (`intact` / `broken`), finetuned on photos of
   wires against several backgrounds/colors.
2. **Bias check**: because the raw dataset confounds label with background in
   places, evaluation includes a background-controlled subset (black-wire-only,
   constant background across both classes) as a genuine bias diagnostic, not
   just the usual validation split.
3. **Export**: PyTorch -> ONNX -> quantized (INT8, w8a8) via Qualcomm AI Hub,
   compiled to `.tflite` targeting the device's Snapdragon core.
4. **Robot loop**: drives forward by default; on 3 consecutive frames
   classified as `broken` above a confidence threshold, stops, and searches
   for the pivot angle that maximizes the classifier's `broken` confidence (the
   nearest available proxy for "wire centered in frame" since this is a
   classifier, not a detector with bounding boxes), photographs, logs, undoes
   the pivot to restore heading, and resumes.

## `wire_finetune/` - training pipeline

| File | Purpose |
|---|---|
| `convert_and_organize.py` | Converts raw HEIC exports into the `intact`/`broken` class folders used for training |
| `train.py` | MobileNetV2 transfer learning: stratified split, heavy augmentation, bias-check eval |
| `export_and_quantize.py` / `continue_quantize.py` | ONNX export + Qualcomm AI Hub quantize/compile job submission (the `continue_*` variant resumes from an already-submitted job id instead of resubmitting) |
| `benchmark_wire_model.py` | Single-image correctness + inference-time check for the quantized `.tflite`, meant to run on-device |
| `camera_stream_server.py` | Standalone MJPEG live-camera server with the classification overlaid, for visually verifying the model before wiring up the robot |
| `runs/` | Training report, and the model at each stage: `.pt` (PyTorch checkpoint), `.onnx` (export), `.tflite` (final quantized, deployed artifact) |

The raw dataset (`dataset/`, `raw_heic/`) is not tracked in git — it's
personal training photos and too large. Regenerate it locally from your own
photo exports with `convert_and_organize.py`.

### Reproducing training

```
pip install torch torchvision pillow pillow-heif qai-hub
python convert_and_organize.py   # raw_heic/ -> dataset/{intact,broken}/...
python train.py                  # -> runs/mobilenet_v2_wire_best.pt, training_report.json
python export_and_quantize.py    # -> runs/wire_classifier.onnx, submits AI Hub quantize+compile job
# if export_and_quantize.py's job polling gets interrupted, resume with:
python continue_quantize.py      # uses the job id already submitted -> runs/wire_classifier_quantized.tflite
```

Qualcomm AI Hub quantize/compile requires an API token (`qai-hub configure --api_token ...`).

## `robot_app/wire-inspector-robot/` - the deployed app

An [Arduino App Lab](https://docs.arduino.cc/software/app-lab/) app: a
Zephyr sketch on the STM32 side does motor control, a Python controller on
the Linux side does camera capture, inference, and the drive/inspect state
machine, connected over the App Lab `Bridge` RPC.

```
app.yaml                app metadata (name, icon, exposed port 8080)
sketch/sketch.ino        motor control, exposes set_motion(mode) over Bridge
sketch/sketch.yaml        board/toolchain profile
python/main.py            camera + inference + state machine + MJPEG monitor
python/requirements.txt   Flask, ai-edge-litert (numpy/opencv come from the
                           device's system Python and are inherited by the
                           app's venv)
```

### Hardware: TB6612FNG motor driver wiring

| Signal | Pin |
|---|---|
| PWMA | 5 |
| AIN1 | 8 |
| AIN2 | 9 |
| PWMB | 6 |
| BIN1 | 10 |
| BIN2 | 11 |
| STBY | 7 |

`set_motion(mode)` — `0` = stop, `1` = forward, `2` = pivot right, `3` = pivot left.
The sketch fails safe: motors are held stopped at boot until Python
explicitly commands motion.

### Hardware: HC-SR04 ultrasonic sensor wiring

| Signal | Pin |
|---|---|
| VCC | 3.3V (**not** 5V - the UNO Q's GPIO is 3.3V logic and isn't confirmed 5V-tolerant; feeding 5V back into a pin risks damaging the board) |
| GND | GND |
| Trig | 3 |
| Echo | 2 |

`get_distance_cm()` returns the measured distance in cm, or `-1.0` if nothing
was in range (or the reading looked physically implausible - e.g. shorter
than the sensor's ~2cm minimum range, which is filtered out as electrical
noise rather than a real echo).

Wired into the drive loop with the same priority as wire inspection, but
obstacle avoidance is checked first each tick since it's about not
colliding with something: on 2 consecutive readings under 15cm, the car
stops, pivots in a fixed direction (a single forward-facing sensor can't
tell which side has more room) until the path reads clear or it gives up
after a bounded number of turns, then resumes. `TESTING_MOTORS_DISABLED` in
`main.py` runs the full detection/decision pipeline with motors kept
silent, for bench-testing sensors without the car driving off.

### Deploying to the device

The App Lab container only bind-mounts the app's own folder, so the model
file has to live inside it (not referenced from elsewhere on the device):

```
# from a machine that can reach the device:
scp -r robot_app/wire-inspector-robot arduino@<device-ip>:/home/arduino/ArduinoApps/

# on the device:
mkdir -p ~/ArduinoApps/wire-inspector-robot/python/models
cp ~/wire_classifier/wire_classifier_quantized.tflite \
   ~/ArduinoApps/wire-inspector-robot/python/models/
# (or copy wire_finetune/runs/wire_classifier_quantized.tflite from this repo)

arduino-app-cli app start user:wire-inspector-robot
```

This compiles and flashes `sketch.ino` to the STM32 and launches
`python/main.py` in a container. Camera index and video device are
hardcoded for a Logitech C615 at `/dev/video2` — adjust `CAMERA_INDEX` in
`main.py` for a different camera.

### Running it

```
arduino-app-cli app start user:wire-inspector-robot   # place the robot before running - it drives immediately
arduino-app-cli app logs user:wire-inspector-robot --follow
arduino-app-cli app stop user:wire-inspector-robot    # always use this to stop, not just closing the browser -
                                                       # it cleanly halts the motors before shutting down
```

Live monitor (label, confidence, robot state, overlaid on the camera feed):
`http://<device-ip>:8080`

Inspection photos and a JSON-lines log of every detected defect are written
to `inspection_log/` inside the app folder on the device (`timestamp`,
`confidence`, `image`, `pivot_ms`).

### Known limitation

The model is a classifier, not an object detector — there's no bounding box
to center on directly. "Centering" is approximated by hill-climbing the
pivot angle that maximizes the model's own `broken` confidence, on the
assumption that a more centered, better-framed wire yields a more confident
classification. It's a reasonable best-effort heuristic for a demo, not a
substitute for real localization.

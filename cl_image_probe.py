import os
os.environ["RUSTICL_ENABLE"] = "adreno,freedreno,fd"
import pyopencl as cl
import numpy as np

platforms = cl.get_platforms()
device = platforms[0].get_devices(cl.device_type.GPU)[0]
ctx = cl.Context([device])

print("Device:", device.name)

def try_image_from_buffer(width, height, label):
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    row_pitch = width * 4 * 4  # width * 4 channels * 4 bytes (float32)
    buf_size = row_pitch * height
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, buf_size)
    try:
        img = cl.Image(ctx, cl.mem_flags.READ_WRITE, fmt,
                        shape=(width, height), pitches=(row_pitch,), buffer=buf)
        print(f"[OK]   {label}: width={width} height={height} row_pitch={row_pitch} (mult of 64: {row_pitch % 64 == 0})")
    except Exception as e:
        print(f"[FAIL] {label}: width={width} height={height} row_pitch={row_pitch} (mult of 64: {row_pitch % 64 == 0}) -> {e}")

# Simula các dạng tensor MobileNetV2 hay gặp, đóng gói kênh vào RGBA
try_image_from_buffer(56, 56, "mid-layer aligned")
try_image_from_buffer(250, 1, "1000-class classifier output (packed/4)")
try_image_from_buffer(320, 1, "1280-channel bottleneck (packed/4)")
try_image_from_buffer(7, 7, "7x7 last conv feature map")
try_image_from_buffer(1, 1, "1x1 global pool output")

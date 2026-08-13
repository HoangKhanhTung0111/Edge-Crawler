import os
os.environ["RUSTICL_ENABLE"] = "adreno,freedreno,fd"
import ctypes

cl = ctypes.CDLL("libOpenCL.so.1")

CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_WRITE = 1 << 0

class cl_image_format(ctypes.Structure):
    _fields_ = [("image_channel_order", ctypes.c_uint32),
                ("image_channel_data_type", ctypes.c_uint32)]

class cl_image_desc(ctypes.Structure):
    _fields_ = [
        ("image_type", ctypes.c_uint32),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
        ("image_depth", ctypes.c_size_t),
        ("image_array_size", ctypes.c_size_t),
        ("image_row_pitch", ctypes.c_size_t),
        ("image_slice_pitch", ctypes.c_size_t),
        ("num_mip_levels", ctypes.c_uint32),
        ("num_samples", ctypes.c_uint32),
        ("mem_object", ctypes.c_void_p),
    ]

CL_MEM_OBJECT_IMAGE2D = 0x10F1
CL_RGBA = 0x10B5
CL_FLOAT = 0x10DE

# get platform/device
num_platforms = ctypes.c_uint32()
cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
platforms = (ctypes.c_void_p * num_platforms.value)()
cl.clGetPlatformIDs(num_platforms.value, platforms, None)
platform = platforms[0]

num_devices = ctypes.c_uint32()
cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(num_devices))
devices = (ctypes.c_void_p * num_devices.value)()
cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices.value, devices, None)
device = devices[0]

err = ctypes.c_int32()
ctx = cl.clCreateContext(None, 1, ctypes.byref(device), None, None, ctypes.byref(err))
print("context err:", err.value)

def try_image_from_buffer(width, height, label, row_pitch=None):
    bytes_per_pixel = 4 * 4  # RGBA float32
    pitch = row_pitch if row_pitch is not None else width * bytes_per_pixel
    buf_size = pitch * height
    buf = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE, ctypes.c_size_t(buf_size), None, ctypes.byref(err))
    if err.value != CL_SUCCESS:
        print(f"[BUF FAIL] {label}: err={err.value}")
        return

    fmt = cl_image_format(CL_RGBA, CL_FLOAT)
    desc = cl_image_desc()
    desc.image_type = CL_MEM_OBJECT_IMAGE2D
    desc.image_width = width
    desc.image_height = height
    desc.image_depth = 0
    desc.image_array_size = 0
    desc.image_row_pitch = pitch
    desc.image_slice_pitch = 0
    desc.num_mip_levels = 0
    desc.num_samples = 0
    desc.mem_object = buf

    img = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, ctypes.byref(fmt), ctypes.byref(desc), None, ctypes.byref(err))
    status = "OK" if err.value == CL_SUCCESS else "FAIL"
    print(f"[{status}] {label}: width={width} height={height} pitch={pitch} (mult64={pitch % 64 == 0}) err={err.value}")

try_image_from_buffer(56, 56, "mid-layer aligned")
try_image_from_buffer(250, 1, "1000-class classifier output packed/4")
try_image_from_buffer(320, 1, "1280-channel bottleneck packed/4")
try_image_from_buffer(7, 7, "7x7 last conv feature map")
try_image_from_buffer(1, 1, "1x1 global pool output")
try_image_from_buffer(250, 1, "1000-class classifier pitch rounded to 64", row_pitch=4032)

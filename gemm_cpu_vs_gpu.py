import os
os.environ["RUSTICL_ENABLE"] = "adreno,freedreno,fd"
import time
import numpy as np
import pyopencl as cl

platforms = cl.get_platforms()
device = platforms[0].get_devices(cl.device_type.GPU)[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

KERNEL = """
__kernel void gemm(
    const int N,
    __global const float *A,
    __global const float *B,
    __global float *C)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""
prg = cl.Program(ctx, KERNEL).build()


def bench_gpu(N):
    a_np = np.random.rand(N, N).astype(np.float32)
    b_np = np.random.rand(N, N).astype(np.float32)
    c_np = np.empty((N, N), dtype=np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    c_g = cl.Buffer(ctx, mf.WRITE_ONLY, c_np.nbytes)

    local_size = 16 if N % 16 == 0 else None
    t0 = time.time()
    evt = prg.gemm(queue, (N, N), (local_size, local_size) if local_size else None,
                   np.int32(N), a_g, b_g, c_g)
    cl.enqueue_copy(queue, c_np, c_g).wait()
    gpu_time = (time.time() - t0) * 1000

    return gpu_time, a_np, b_np, c_np


def bench_cpu(N, a_np, b_np):
    t0 = time.time()
    c_cpu = a_np @ b_np
    cpu_time = (time.time() - t0) * 1000
    return cpu_time, c_cpu


print(f"{'N':>6} | {'GPU (ms)':>10} | {'CPU (ms)':>10} | {'ty le GPU/CPU':>14} | {'max diff':>10}")
print("-" * 65)
for N in [256, 512, 1024, 1536, 2048]:
    gpu_time, a_np, b_np, c_gpu = bench_gpu(N)
    cpu_time, c_cpu = bench_cpu(N, a_np, b_np)
    diff = np.abs(c_gpu - c_cpu).max()
    ratio = gpu_time / cpu_time
    faster = "GPU nhanh hon" if ratio < 1 else ""
    print(f"{N:>6} | {gpu_time:>10.2f} | {cpu_time:>10.2f} | {ratio:>13.2f}x | {diff:>10.4f} {faster}")

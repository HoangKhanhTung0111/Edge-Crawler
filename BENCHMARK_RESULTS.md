# CPU/GPU Inference Benchmark Results

Measured on the Arduino UNO Q (Qualcomm Dragonwing QRB2210, Adreno 702 GPU),
comparing 7 CNN architectures across CPU and GPU, float vs quantized, and
two different runtimes — as part of choosing the architecture for the wire
defect classifier.

## 1. CPU: 7 architectures, float vs quantized

| Model | Runtime | Float (ms) | Quantized (ms) | Ratio | Quantized size |
|---|---|---|---|---|---|
| **MobileNetV2** | TFLite + XNNPACK | ~90 ms | **23.72 ms** ⚡ | ~3.8x faster | 3.8 MB |
| ResNet18 | TFLite + XNNPACK | 131.56 ms | 82.24 ms | 1.60x faster | 11.87 MB |
| SqueezeNet1.1 | TFLite + XNNPACK | 28.51 ms | 19.85 ms | 1.44x faster | 1.36 MB |
| ShuffleNet-v2 | TFLite + XNNPACK | 8.63 ms | 6.49 ms | 1.33x faster | 1.59 MB |
| MobileNetV3-Small | ONNX Runtime | 28.70 ms | 52.29 ms 🔻 | 0.55x (**slower**) | 10.43 MB |
| EfficientNet-B0 | ONNX Runtime | 124.88 ms | 237.36 ms 🔻 | 0.53x (**slower**) | 21.76 MB |
| EfficientNet-V2-S | ONNX Runtime | 1237.43 ms | 1613.73 ms 🔻 | 0.77x (**slower**) | 87.52 MB |

**Key finding: quantization doesn't automatically mean faster.** The last
three models aren't supported by Qualcomm AI Hub's TFLite export path, only
ONNX — and `onnxruntime`'s default CPU execution provider has no
accelerated INT8/UINT16 kernels for ARM, so quantized inference there pays
a dequantize→compute→requantize tax with nothing to offset it, ending up
*slower* than float. Only the first four models, which went through the
**TFLite + XNNPACK** pair specifically, actually benefited from
quantization — XNNPACK has real NEON-accelerated quantized kernels.

## 2. MobileNetV2: CPU vs GPU

| | CPU (XNNPACK + quantized) | GPU (Vulkan/ncnn) |
|---|---|---|
| MobileNetV2 | **23.72 ms** | 374 ms (15.8x slower) |

GPU only wins on raw matrix multiplication (GEMM 256×256: GPU 76.1ms vs
CPU 145.2ms) — not on real CNN inference, where per-layer dispatch overhead
dominates at this model size.

## 3. Final deployed model (wire defect classifier, 2-class finetune)

Not the generic 1000-class ImageNet weights above — this is the actual
finetuned, quantized, deployed artifact:

| File | Size |
|---|---|
| `mobilenet_v2_wire_best.pt` (PyTorch checkpoint) | 9.15 MB |
| `wire_classifier.onnx` (export) | 8.88 MB |
| `wire_classifier_quantized.tflite` (deployed) | **2.72 MB** |

## Why MobileNetV2, not ShuffleNet-v2 (the fastest at 6.49ms)?

Three reasons, in order of importance:

1. **The 17ms difference is irrelevant to this application.** The robot
   moves slowly and the decision loop already has far larger delays built
   in (e.g. a 250ms settle time per pivot step while scanning for a broken
   wire). Neither 23.72ms nor 6.49ms is anywhere near a real bottleneck —
   there's no 30fps real-time constraint here, so "faster" buys nothing
   practically.

2. **Capacity risk for a fine-grained task.** ShuffleNet-v2's speed comes
   from aggressive channel shuffling and grouped convolutions at only
   ~1.4M params — a deliberate trade of representational capacity for
   compute efficiency. Telling an intact wire from a broken one is a
   *subtle* visual distinction (a small gap in a thin object), not a
   coarse category split — exactly the kind of task where a
   capacity-starved architecture is more likely to underperform, and with
   only ~330 training images there isn't much data to compensate for a
   weaker feature extractor.

3. **Only MobileNetV2 was actually verified end-to-end on the real task.**
   The 6.49ms figure for ShuffleNet-v2 is from generic 1000-class ImageNet
   classification — it was never finetuned, bias-checked, quantized, or
   deployed on the actual wire dataset. MobileNetV2 was taken all the way
   through that full pipeline and hit 100% validation accuracy *and* 100%
   accuracy on the background-controlled bias-check subset (black wire
   only, constant background across both classes) on real hardware. A
   known-good, fully-verified result beats an untested hypothesis about a
   faster architecture, especially this close to a demo.

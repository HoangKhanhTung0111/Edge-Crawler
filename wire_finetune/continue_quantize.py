import json
from pathlib import Path
import qai_hub as hub

BASE = Path(r"C:\Users\Admin\unoq\wire_finetune")
RUNS = BASE / "runs"

qjob = hub.get_job("jp4378zq5")
print(f"Quantize job: {qjob.job_id} -> {qjob.url}")
qjob.wait()
print("Quantize status:", qjob.get_status())

quantized_model = qjob.get_target_model()
print("Got quantized model:", quantized_model)

device = hub.Device("Snapdragon 8 Elite QRD")
print("Submitting compile job (tflite)...")
cjob = hub.submit_compile_job(
    quantized_model,
    device=device,
    name="wire_classifier_tflite_w8a8",
    options="--target_runtime tflite",
)
print(f"Compile job: {cjob.job_id} -> {cjob.url}")
cjob.wait()
print("Compile status:", cjob.get_status())

target_model = cjob.get_target_model()
out_path = RUNS / "wire_classifier_quantized.tflite"
target_model.download(str(out_path))
print(f"Downloaded quantized tflite -> {out_path}")

with open(RUNS / "quantize_job_info.json", "w") as f:
    json.dump({
        "quantize_job_id": qjob.job_id,
        "quantize_job_url": qjob.url,
        "compile_job_id": cjob.job_id,
        "compile_job_url": cjob.url,
    }, f, indent=2)

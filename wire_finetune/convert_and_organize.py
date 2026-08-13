import os
import shutil
from pathlib import Path
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

BASE = Path(r"C:\Users\Admin\unoq\wire_finetune")
RAW = BASE / "raw_heic"
OUT = BASE / "dataset"

SOURCES = {
    "lanh_vang": (RAW / "lanh_vang" / "Data", "lanh", "vang"),
    "dut_vang": (RAW / "dut_vang" / "Data dây đứt ", "dut", "vang"),
    "dut_den": (RAW / "dut_den" / "Đứt", "dut", "den"),
    "lanh_den": (RAW / "lanh_den" / "Lành v2", "lanh", "den"),
}

MAX_SIDE = 800  # resize longest side to this, keep aspect ratio; plenty for 224x224 training crops

for name, (src_dir, label, color) in SOURCES.items():
    out_dir = OUT / label / f"{color}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    heic_files = sorted(src_dir.glob("*.HEIC")) + sorted(src_dir.glob("*.heic"))
    n_ok, n_fail = 0, 0
    for f in heic_files:
        try:
            img = Image.open(f).convert("RGB")
            w, h = img.size
            scale = MAX_SIDE / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            out_path = out_dir / (f.stem + ".jpg")
            img.save(out_path, "JPEG", quality=92)
            n_ok += 1
        except Exception as e:
            print(f"FAILED: {f.name}: {e}")
            n_fail += 1
    print(f"{name}: {n_ok} ok, {n_fail} failed -> {out_dir}")

print("\nTong ket:")
for label in ["lanh", "dut"]:
    total = sum(1 for _ in (OUT / label).rglob("*.jpg"))
    print(f"  {label}: {total} anh")

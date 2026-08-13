import numpy as np
from PIL import Image

BASE = "/home/arduino/detectron2_test"

img = Image.open(f"{BASE}/../gpu_test/test_image.jpg").convert('RGB')
img = img.resize((800, 800), Image.Resampling.BILINEAR)
arr = np.array(img, dtype=np.float32) / 255.0  # value_range [0,1]
arr = np.expand_dims(arr, axis=0)  # (1,800,800,3) NHWC per metadata
print("shape:", arr.shape, "min/max:", arr.min(), arr.max())

arr.tofile(f"{BASE}/image_800.raw")

with open(f"{BASE}/raw_list_proposal.txt", "w") as f:
    f.write("image_800.raw\n")

print("Saved image_800.raw and raw_list_proposal.txt")

from pillow_heif import register_heif_opener
from PIL import Image
import os

register_heif_opener()  
input_dir = "camera_input_pic"
output_dir = "camera_input_pic_converted"
os.makedirs(output_dir, exist_ok=True)

heic_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".heic")])

for i, file in enumerate(heic_files):
    img = Image.open(os.path.join(input_dir, file))
    out_path = os.path.join(output_dir, f"{i:03d}.jpg")
    img.save(out_path, "JPEG")

print("Done.")

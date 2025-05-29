from pillow_heif import register_heif_opener
from tqdm import tqdm
from PIL import Image
import os
import argparse

def convert(input_dir):
    register_heif_opener()  
    output_dir = "camera_input_pics_converted"
    os.makedirs(output_dir, exist_ok=True)

    camera_input_pics = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".heic", ".jpg", '.jpeg'))
    ])

    for i, file in enumerate(tqdm(camera_input_pics, desc="Converting images")):
        img = Image.open(os.path.join(input_dir, file))
        out_path = os.path.join(output_dir, f"{i:03d}.jpg")
        img.save(out_path, format="JPEG", quality=95)

    print("Image converted.")


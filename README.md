# README

---

## Preprocessing
### 1. Environment Setup: Nerfstudio and COLMAP
```bash 
conda create -n nerfstudio_env -c conda-forge python=3.10
conda activate nerfstudio_env
pip install nerfstudio
conda install -c conda-forge colmap
conda install -c conda-forge ffmpeg
```

### 2. Install GSplat Extension
```bash 
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
```

### 3. GPU Build Configuration
```bash 
export MAX_JOBS=1
```
#### Optional: Set GPU Architecture
```bash 
export TORCH_CUDA_ARCH_LIST="8.9" # for RTX 4090
export TORCH_CUDA_ARCH_LIST="8.0" # for A100
```

```bash 
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=1
```

### 4. Convert Camera Input Images (Optional HEIC to PNG)
```bash 
pip install pillow-heif
python convert.py
```

### 5. Generate Camera Poses with COLMAP
```bash
ns-process-data images --data ./camera_input_images_converted --output-dir ./processed_images_colmap
```

### 6. Train with Splatfacto
```bash
ns-train splatfacto --data ./processed_images_colmap
```

### 7. Dump out the Gaussian Splat
- Export ply file
```bash
ns-export gaussian-splat --load-config logs/splatfacto/<time>/config.yml --output-dir ./export/splat
```
- Rename ply file
```bash
mv export/splat/splat.ply export/splat/rename.ply
```

### 8. Link to Download ply File
- Put `nubzuki.ply` under data/ directory
```bash
https://drive.google.com/drive/folders/1U4meVGaYqIFF0W6BxDdylooCOpICI9cx?usp=sharing
```

### 9. Link to Download Preprocessed File
- The camera input pictures file is gitignored: `camera_input_pic`, `camera_input_pic_convert`
- The preprocessed file is gitignored: `processed_images_colmap`
```bash
https://drive.google.com/drive/folders/15lzamNo2JjFHmjq44iJfDnQnInIL363u?usp=sharing
```

---

## Mirror Rendering
### 1. Activate Conda Environment Same as CS479 Assignment 3
```bash
conda activate cs479-gs
```

### 2. Render the Scene
```bash
python render.py
```

---

# TODO
1. [x] Adapt the cut of the scene to the camera angle
2. [x] Make the mirror in the scene
3. [x] Find the camera path
4. [x] Make the inception effect
5. [x] Take new images (Mon.)

<div align=center>
  <h1>
    Mirror Rendering of Gaussian Splat
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs479-spring-2025/ target="_blank"><b>KAIST CS479: Machine Learning for 3D Data</b></a><br>
    Group Project
  </p>
</div>

## Code Structure
This codebase is organized as the following directory tree.
```
Rendering-Contest
│
├── camera_input_images
├── camera_input_images_converted
├── data
├── export
├── mirror_rendering_outputs
├── outputs
├── processed_images_colmap
├── src
│   ├── camera.py
│   ├── constants.py
│   ├── renderer.py
│   ├── rgb_metrics.py
│   ├── scene.py
│   └── sh.py
├── .gitignore
├── convert.py
├── evaluate.py
├── metrics.csv
├── render.py
├── render_all.sh
└── README.md
```

## Preprocessing

### 1. Environment Setup
```bash 
conda create -n nerfstudio_env -c conda-forge python=3.10 -y
conda activate nerfstudio_env
pip install nerfstudio
conda install -c conda-forge colmap ffmpeg -y
conda install \
  pytorch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  pytorch-cuda=11.8 \
  -c pytorch -c nvidia \
  -y
```

### 2. Install GSplat Extension
```bash 
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
```

### 3. GPU Build Configuration
#### Set Max Job
```bash 
export MAX_JOBS=1
```

#### Set GPU Architecture
##### Option 1: for RTX 4090
```bash 
export TORCH_CUDA_ARCH_LIST="8.9"
```

##### Option 2: for A100
```bash 
export TORCH_CUDA_ARCH_LIST="8.0"
```

##### Option 3: for A6000
```bash 
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TORCH_CUDA_ARCH_LIST="8.6"
```

### 4. Convert Camera Input Images (Optional HEIC to PNG)
```bash 
pip install pillow-heif
python convert.py
```
- The camera input pictures are gitignored: `camera_input_pic/`, `camera_input_pic_convert/`
- Link to Download: https://drive.google.com/drive/folders/1uLroHJXeJLAx3mO67CzmIuwsV-WOKOWP?usp=sharing

### 5. Generate Camera Poses with COLMAP
```bash
ns-process-data images --data ./camera_input_pic_converted --output-dir ./processed_images_colmap
```
- The COLMAP–processed images are gitignored: `processed_images_colmap/`
- Link to Download: https://drive.google.com/drive/folders/15lzamNo2JjFHmjq44iJfDnQnInIL363u?usp=sharing

### 6. Train with Splatfacto
```bash
ns-train splatfacto --data ./processed_images_colmap
```
- The Splatfacto trained model outputs are gitignored: `outputs/`
- Link to Download: https://drive.google.com/drive/folders/1AniBSBACpUI5WL0LCbVa4b_IqVVTidcM?usp=sharing

### 7. Dump out the Gaussian Splat
- Export ply file
```bash
ns-export gaussian-splat \
  --load-config outputs/processed_images_colmap/splatfacto/{timestamp}/config.yml \
  --output-dir ./export/splat
```
- Rename ply file
```bash
mv export/splat/splat.ply export/splat/{rename}.ply
```
- The Gaussian Splat is gitignored: `export/splat/`
- Link to Download: https://drive.google.com/drive/folders/1U4meVGaYqIFF0W6BxDdylooCOpICI9cx?usp=sharing

## Mirror Rendering

### 1. Activate Conda Environment Same as CS479 Assignment 3
```bash
conda deactivate nerfstudio_env
conda activate cs479-gs
```

#### cs479-gs Environment Setup (Optional)
```bash
conda create --name cs479-gs python=3.10
conda activate cs479-gs
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torchmetrics[image]
pip install imageio[ffmpeg]
pip install plyfile tyro==0.6.0 jaxtyping==0.2.36 typeguard==2.13.3
pip install simple-knn/.
```

#### PyTorch version might be incompatible with system CUDA version (Optional)
- Check CUDA version installed and verify PyTorch CUDA Version compatibility
```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
- Downgrade CUDA To 12.1
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Render the Scene
```bash
python render.py
```
- The mirror rendering output file is gitignored: `mirror_rendering_outputs/`
- Link to Download: https://drive.google.com/drive/folders/1ZRAbBIHspBpg4I_Ix_4AJKGpS3aoricH?usp=sharing

---

# TODO
1. [x] Adapt the cut of the scene to the camera angle
2. [x] Make the mirror in the scene
3. [x] Find the camera path
4. [x] Make the inception effect
5. [x] Take new images (Mon.)

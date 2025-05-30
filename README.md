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
Important: It will only look like this after all uncessecary folders are removed (running preprocess.py)
```
Rendering-Contest
│
├── camera_input_images (Your Camera input files)
├── data
│   ├── nubuzuki_only_v2
│   |   └── nubuzuki_only_v2.json
|   └── nubuzuki_only_v2.ply 
├── rendering_outputs
├── src
│   ├── camera.py
│   ├── constants.py
│   ├── renderer.py
│   ├── rgb_metrics.py
│   ├── scene.py
│   └── sh.py
├── .gitignore
├── convertor.py
├── render.py
├── preprocess.py
├── path_creator.py
└── README.md
```

## Preprocessing
- If you only want to render the final output please go to section: Mirror Rendering 
### 1. Environment Setup
```bash 
conda create -n nerfstudio_env -c conda-forge python=3.10 -y
conda activate nerfstudio_env
pip install nerfstudio
pip install pillow-heif
pip install tqdm
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
conda install -c conda-forge colmap -y
conda install -c conda-forge ffmpeg -y
conda install \
  pytorch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  pytorch-cuda=11.8 \
  -c pytorch -c nvidia \
  -y
```
### Preprocessing (automated): 
Script only supports GPU: RTX490, A100, A6000
Scene_name for our scene is: nubzuki_only_v2
input_dir you can download the input pictures converted or unconverted: https://drive.google.com/drive/folders/1zehi2jmguVz13y1qFWzGgW9K9I2LFjAj
```bash 
  python preprocess.py  --remove_all --convert --colmap --train --ply --scene_name "<YOUR_SCENE_NAME>" --input_dir "<PATH_TO_YOUR_FOLDER>" --GPU "<GPU_NAME>"
``` 

### Preprocessing (Step-by-step):
### 2. Convert Camera Input Images (Optional HEIC to PNG)
```bash 
python convertor.py
```
- The camera input pictures are gitignored: `camera_input_pics/`, `camera_input_pics_converted/`
- Link to Download: https://drive.google.com/drive/folders/1uLroHJXeJLAx3mO67CzmIuwsV-WOKOWP?usp=sharing

### 3. Generate Camera Poses with COLMAP
```bash
ns-process-data images --data ./camera_input_pics_converted --output-dir ./processed_images_colmap
```
- The COLMAP–processed images are gitignored: `processed_images_colmap/`
- Link to Download: https://drive.google.com/drive/folders/15lzamNo2JjFHmjq44iJfDnQnInIL363u?usp=sharing

### 4. GPU Build Configuration

#### Option 1: for RTX 4090
```bash 
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.9"
```

#### Option 2: for A100
```bash 
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.0"
```

#### Option 3: for A6000
```bash 
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.6"
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

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
- Copy ply file to `data/` directory
```bash
cp export/splat/{rename}.ply data/
```

## Mirror Rendering

### 1. Activate Conda Environment Same as CS479 Assignment 3
```bash
conda deactivate
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

### 2. Scene Path:
```bash
python path_creator.py
```


### 3. Render the Scene
```bash
python render.py
```
- The mirror rendering output file is gitignored: `mirror_rendering_outputs/`
- Link to Download: https://drive.google.com/drive/folders/1ZRAbBIHspBpg4I_Ix_4AJKGpS3aoricH?usp=sharing

---


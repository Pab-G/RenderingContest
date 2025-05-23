# README

## 1. Environment Setup: Nerfstudio and COLMAP
```bash 
conda create -n nerfstudio_env -c conda-forge python=3.10
conda activate nerfstudio_env
pip install nerfstudio
conda install -c conda-forge colmap
conda install -c conda-forge ffmpeg
```

## 2. Install GSplat Extension
```bash 
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
```

## 3. GPU Build Configuration
```bash 
export MAX_JOBS=1
```
### Optional: Set GPU Architecture
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

## 4. Convert Camera Input Images (Optional HEIC to PNG)
```bash 
pip install pillow-heif
python convert.py
```

## 5. Generate Camera Poses with COLMAP
```bash
ns-process-data images --data ./camera_input_images_converted --output-dir ./processed_images_colmap
```

## 6. Train with Splatfacto
```bash
ns-train splatfacto --data ./processed_images_colmap
```

## TODO
```bash
1. Adapt the cut of the scene to the camera angle
2. Make the mirror in the scene
3. Find the camera path
4. Make the inception effect
5. Take new images (Mon.)
```





## Installation:

### Nerfstudio and colmap
```bash 
conda create -n nerfstudio_env -c conda-forge python=3.10
conda activate nerfstudio_env
pip install nerfstudio
conda install -c conda-forge colmap
conda install -c conda-forge ffmpeg
```
### install Splatfacto
```bash 
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
```

### Image converter from heic to jpg
```bash 
pip install pillow-heif
```

### Folder organisation
```bash 
in data put the splath form training and renamed it chair
```

### Change some config
```bash 
export MAX_JOBS=1
```

### GILLE
```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

### PABLO
```bash
export TORCH_CUDA_ARCH_LIST="8.0"
```

### Custom scene
```bash
ns-process-data images --data ./data/images --output-dir outputs/custom_scene
ns-train splatfacto --data outputs/custom_scene/
```






## Installation:

### Nerfstudio qnd colmap
conda create -n nerfstudio_env -c conda-forge python=3.10
conda activate nerfstudio_env
pip install nerfstudio
conda install -c conda-forge colmap

### install Splatfacto
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0


### Image converter fro, heic to jpg
pip install pillow-heif

### Change some config
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.9"
(depend of the gpu)

### Custom scene
ns-process-data images --data ./data/images --output-dir outputs/custom_scene
ns-train splatfacto --data outputs/custom_scene/







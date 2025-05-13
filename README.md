## Installation:

### Nerfstudio qnd colmap
conda create -n nerfstudio_env -c conda-forge python=3.10
conda activate nerfstudio_env
pip install nerfstudio
conda install -c conda-forge colmap

### install Splatfacto
pip install gsplat

### Image converter fro, heic to jpg
pip install pillow-heif

### Custom scene
ns-process-data images --data ./data/images --output-dir outputs/custom_scene
ns-train splatfacto --data outputs/custom_scene/
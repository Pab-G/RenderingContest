import argparse
import os
import shutil
from convertor import convert
import subprocess
import time
import signal

SPLATFACTO_PATH = "outputs/processed_images_colmap/splatfacto"

def get_timestamp():
    output_base = SPLATFACTO_PATH
    subdirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
    latest = max(subdirs)
    return latest

def main(args):
    os.environ["MAX_JOBS"] = "1"
    if args.GPU == "A100":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    elif args.GPU == "RTX4090":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    elif args.GPU == "A6000":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
        os.environ["PATH"] = f"/usr/local/cuda-12.2/bin:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda-12.2/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"    
    else: 
        raise ValueError(f"Invalid GPU: {args.GPU}")
    
    if args.convert:
        convert(args.input_dir) 
    
    if args.colmap:
        try: 
            subprocess.run([
                "ns-process-data",
                "images",
                "--data", "./camera_input_pics_converted",
                "--output-dir", "./processed_images_colmap"
            ], check=True, text=True, stdout=None, stderr=None)
        except subprocess.CalledProcessError as e:
            print(f"Error running COLMAP: {e}")
            raise e
        
    if args.train:
        print("Will now train the Splats..")
        try: 
            proc = subprocess.Popen(
                ["ns-train", "splatfacto", "--data", "./processed_images_colmap"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in proc.stdout:
                print(line, end="")
                if "Training Finished" in line:
                    time.sleep(2)
                    proc.send_signal(signal.SIGINT)
                    break
            proc.wait()
        except subprocess.CalledProcessError as e:
            print(f"Error while training Splats: {e}")
            raise e
        print("Splats trained successfully.")
    
    if args.ply:
        print("Will now create the PLY file..")
        timestamp = get_timestamp()
        try: 
            subprocess.run([
                "ns-export", "gaussian-splat",
                "--load-config", f"./outputs/processed_images_colmap/splatfacto/{timestamp}/config.yml",
                "--output-dir", "./data"
            ], check=True, text=True, stdout=None, stderr=None)
        except subprocess.CalledProcessError as e:
            print(f"Error while creating PLY file: {e}")
            raise e
        print("PLY file created successfully.")
        print("Will now change the folder structure correctly..")
        os.rename("./data/splat.ply", f"./data/{args.scene_name}.ply")

    if args.remove_all:
        print("Will now remove all created folders and leave only the ply file..")
        # Remove old image folder
        if os.path.exists("camera_input_pics_converted"):
            shutil.rmtree("camera_input_pics_converted")
            print("Removed camera_input_pics_converted folder")
        if os.path.exists("processed_images_colmap"):
            shutil.rmtree("processed_images_colmap")
            print("Removed processed_images_colmapfolder")
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")
            print("Removed outputs folder")
        if os.path.exists(f"{args.input_dir}"):
            shutil.rmtree(f"{args.input_dir}")
            print(f"Removed {args.input_dir} folder")
    print("Done!")
    
    
if __name__ == "__main__":
    ################################################################################################################################################################
    # If you just have gotten the folder with .heic images, run:
    # python preprocess.py --GPU <GPU_NAME> -- remove_all--convert --colmap --train --ply --scene_name "<YOUR_SCENE_NAME>" --input_dir "<PATH_TO_YOUR_FOLDER>"
    ################################################################################################################################################################
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="To preprocess:")
    parser.add_argument("--GPU", type=str, default="A100", help= "GPU you are using, default is A100")
    parser.add_argument("--train", action='store_true', default=False, help= "Whether to train the Splats, default is False")
    parser.add_argument("--ply", action='store_true', default=False, help= "Whether you already have the PLY file, default is False")
    parser.add_argument("--scene_name", type=str, default="nubzuki_only_v2", help= "Name of the scene, default is nubzuki_only_v2")
    parser.add_argument("--input_dir", type=str, default="camera_input_pics", help= "Path to the input directory, default is camera_input_pics")
    parser.add_argument("--remove_all", action='store_true', default=False, help= "Whether to remove created, not needed folders, default is False")
    parser.add_argument("--convert", action='store_true', default=False, help= "Whether to convert the images from HEIC to PNG, default is True")
    parser.add_argument("--colmap", action='store_true', default=False, help= "Whether to run COLMAP, default is False")
    args = parser.parse_args()
    main(args)
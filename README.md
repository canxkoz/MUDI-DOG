# MUDI-DOG

This repository contains the `infer.py` script used for running inferences on a single image using Wonder3D, Yolo-world, Dino, Magicpony and can generate 3D objects.

## Overview

The `infer.py` script allows users to perform inference on input images with various configurations and settings. The script uses Wonder3d, Yolo-world, Dino, Magicpony and can generate 3D objects.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/canxkoz/MUDI-DOG.git
    cd MUDI-DOG
    ```

2. Create a Conda environment and install the required packages:

    ```bash
    conda create -n mudidog python=3.10
    conda activate mudidog
    ```

3. Install the required packages for Wonder3D and Yolo-world:

    ```bash
    pip install -r requirements.txt
    ```

4. Install the required packages for MagicPony:

    ```bash
    pip install git+https://github.com/NVlabs/nvdiffrast/
    pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn@v1.6#subdirectory=bindings/torch
    imageio_download_bin freeimage

5. Install PyTorch3D

    ```bash
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    ```

## Usage

To run the inference script, use the following command:

```bash
python infer.py --config Wonder3D/configs/mvdiffusion-joint-ortho-6views.yaml --input_path examples/horse.jpg --output_path magicpony_inputs --use_dino_features --view_count 3 --object horse

```
## Command Line Arguments:

- `--config`: The path to the configuration file for Wonder3D.
- `--input_path`: The path to the input image for feeding into the Wonder3D.
- `--output_path`: The path to save the outputs for the magicpony.
- `--use_dino_features`: Flag to indicate whether to use DINO features when feeding views into magicpony. No argument needed, presence of the flag means it will be used.
- `--view_count`: Number of views to generate or consider during the inference process. This number of views will be used to feed into the magicpony.
- `--object`: The object to generate 3D objects for. (horse, bird)


## Important Notes

- for use magicpony, you need to install tets to `MagicPony/data/tets` folder with;
    ```bash
    wget https://download.cs.stanford.edu/viscam/AnimalKingdom/magicpony/data/tets.zip && unzip -q tets.zip
    ```

- Also you need to install the pretrained models to `MagicPony/checkpoints/object_name` folder from the link below;
    ```bash
    https://drive.google.com/file/d/1zpmkPGq5Gc0T5FF5EUfR1mjVmBIWhKPh/view?usp=sharing
    ```

- You can generate horses using the following model;
     ```bash
    https://drive.google.com/file/d/1Qpp1wNIQGjOY9r0mfb5iY-JltwL96J_r/view?usp=sharing
    ```

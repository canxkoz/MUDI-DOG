# MUDİ-DOG

This repository contains the `infer.py` script used for running inferences on a single image using Wonder3D, Yolo-world, Dino, Magicpony and can generate 3D objects.

## Overview

The `infer.py` script allows users to perform inference on input images with various configurations and settings. The script uses Wonder3d, Yolo-world, Dino, Magicpony and can generate 3D objects.

## Requirements

this section will be added soon.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/canxkoz/MUDI-DOG.git
    cd MUDI-DOG
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt (this section will be added soon)
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

- Our Magicpony checkpoints will added to the repository soon.

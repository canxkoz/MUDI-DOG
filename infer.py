from MudiDog.config import TestConfig
from MudiDog.utils import load_wonder3d_pipeline, sam_init, wonder3d_preprocess, run_wonder3d_pipeline, get_bounding_box, extract_mask_from_img

from Wonder3D.utils.misc import load_config

from omegaconf import OmegaConf

from PIL import Image
import numpy as np
import torch

import uuid

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    # parse YAML config to OmegaConf
    cfg = load_config("/YOUR-HOME-PATH/Wonder3D/configs/mvdiffusion-joint-ortho-6views.yaml")
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    pipeline = load_wonder3d_pipeline(cfg)
    torch.set_grad_enabled(False)
    pipeline.to(f'cuda:{0}') # move to GPU

    # prepare the input image for the pipeline
    predictor = sam_init()

    OBJECT_NAME = "bird"

    orj_image_path = f"/YOUR-HOME-PATH/MUDI-DOG/examples/{OBJECT_NAME}.png" # path to the original image
    output_path = "/YOUR-HOME-PATH/MUDI-DOG/magicpony_inputs" # path to save the views

    # load original image
    orj_image = Image.open(orj_image_path).convert("RGB")

    # preprocess the image for wonder3d
    input_image_320 = wonder3d_preprocess(predictor, orj_image, segment=True, rescale=False)

    # run the wonder3d pipeline
    view_1, view_2, view_3, view_4, view_5, view_6, \
    normal_1, normal_2, normal_3, normal_4, normal_5, normal_6, \
    _, _ = run_wonder3d_pipeline(pipeline, cfg, input_image_320, 3, 50, 42, 256)

    views = [view_2, view_3, view_4, view_5, view_6]

    # resize the original image to 256x256
    orj_view = orj_image.resize((256, 256))

    # extract mask from the original image
    orj_view_without_bg = np.array(input_image_320.resize((256, 256)))
    orj_view_without_bg = np.where(orj_view_without_bg == 0, 255, orj_view_without_bg)
    mask = extract_mask_from_img(orj_view_without_bg)

    # extract bounding box from the mask
    bbox = get_bounding_box(mask)

    image_id = str(uuid.uuid4())

    # save the original image
    orj_view.save(f"{output_path}/{image_id}_rgb.png")
    Image.fromarray(mask).save(f"{output_path}/{image_id}_mask.png")

    with open(f"{output_path}/{image_id}_bbox.txt", "w") as f:
            f.write(f"{image_id}" + " ")
            f.write(" ".join([str(i) for i in bbox]))
            f.write(" " + str(-1))

    # extract same masks and bounding boxes from the views
    for i, view in enumerate(views):
        mask = extract_mask_from_img(view)
        bbox = get_bounding_box(mask)

        Image.fromarray(view).save(f"{output_path}/{image_id}_view_{i+1}.png")
        Image.fromarray(mask).save(f"{output_path}/{image_id}_view_{i+1}_mask.png")

        with open(f"{output_path}/{image_id}_view_{i+1}_bbox.txt", "a") as f:
            f.write(str(f"{image_id}_view_{i+1}") + " ")
            f.write(" ".join([str(i) for i in bbox]))
            f.write(" " + str(-1))

    


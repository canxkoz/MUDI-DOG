from MudiDog.config import TestConfig
from MudiDog.utils import load_wonder3d_pipeline, sam_init, wonder3d_preprocess, run_wonder3d_pipeline

from Wonder3D.utils.misc import load_config

from omegaconf import OmegaConf

from PIL import Image
import torch

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
    output_path = "/YOUR-HOME-PATH/MUDI-DOG/outputs" # path to save the views

    # load original image
    orj_image = Image.open(orj_image_path).convert("RGB")

    # preprocess the image for wonder3d
    input_image_320 = wonder3d_preprocess(predictor, orj_image, segment=True, rescale=False)

    # run the wonder3d pipeline
    view_1, view_2, view_3, view_4, view_5, view_6, \
    normal_1, normal_2, normal_3, normal_4, normal_5, normal_6, \
    _, _ = run_wonder3d_pipeline(pipeline, cfg, input_image_320, 3, 50, 42, 256)
    views = [view_1, view_2, view_3, view_4, view_5, view_6]

    # save the views
    for i, view in enumerate(views):
        view_image = Image.fromarray(view)
        view_image.save(f"{output_path}/{OBJECT_NAME}_view_{i}.png")
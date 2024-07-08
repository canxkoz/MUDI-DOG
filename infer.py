from MudiDog.config import TestConfig
from MudiDog.utils import load_wonder3d_pipeline, sam_init, wonder3d_preprocess, run_wonder3d_pipeline, get_bounding_box, extract_mask_from_img

from Wonder3D.utils.misc import load_config

from omegaconf import OmegaConf

from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

import os
import uuid
import argparse

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="/YOUR-HOME-PATH/MUDI-DOG/Wonder3D/configs/mvdiffusion-joint-ortho-6views.yaml")
    args.add_argument("--examples_path", type=str, default="/YOUR-HOME-PATH/MUDI-DOG/examples")
    args.add_argument("--output_path", type=str, default="/YOUR-HOME-PATH/MUDI-DOG/magicpony_inputs")
    args.add_argument("--view_count", type=int, default=3)
    args.add_argument("--object", type=str, default="bird")
    args = args.parse_args()

    config = args.config
    examples_path = args.examples_path
    output_path = args.output_path
    object_name = args.object
    view_count = args.view_count

    # if output path does not exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # device
    device= "cuda" if torch.cuda.is_available() else "cpu"

    # load the YOLO-World model
    model = YOLO('yolov8x-worldv2.pt')  # or choose yolov8m/l-world.pt
    classes = ["others", object_name] # the class that we are interested in, in this case it is birds
    model.set_classes(classes)

    
    # parse YAML config to OmegaConf
    cfg = load_config(args.config)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    pipeline = load_wonder3d_pipeline(cfg)
    torch.set_grad_enabled(False)
    pipeline.to(device) # move to GPU

    # prepare the input image for the pipeline
    predictor = sam_init()

    orj_image_path = f"{examples_path}/{object_name}.png" # path to the original image

    # load original image
    orj_image = Image.open(orj_image_path).convert("RGB")

    # preprocess the image for wonder3d
    input_image_320 = wonder3d_preprocess(predictor, orj_image, segment=True, rescale=False)

    # run the wonder3d pipeline
    view_1, view_2, view_3, view_4, view_5, view_6, \
    normal_1, normal_2, normal_3, normal_4, normal_5, normal_6, \
    _, _ = run_wonder3d_pipeline(pipeline, cfg, input_image_320, 3, 50, 42, 256)

    views = [view_2, view_3, view_4, view_5, view_6]

    index_score_dict = {}
    for i, view in enumerate(views):
        results = model.predict(view, verbose=False, device=device)
        for j, c in enumerate(results[0].boxes.cls):
            if c.cpu().numpy() == 1.0 and results[0].boxes.conf.cpu().numpy()[j] > 0.1:
                if i in index_score_dict and index_score_dict[i] < results[0].boxes.conf.cpu().numpy()[j]:
                    continue
                index_score_dict[i] = results[0].boxes.conf.cpu().numpy()[j]

    if len(index_score_dict) < 3:
        raise ValueError("The object is not detected in at least 3 views, please try again with a different image.")
    
    if view_count == 3:
        # get the 3 views with the highest confidence
        index_score_dict = dict(sorted(index_score_dict.items(), key=lambda x: x[1], reverse=True)[:3])
        new_views = []
        for item, _ in index_score_dict.items():
            new_views.append(views[item])

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

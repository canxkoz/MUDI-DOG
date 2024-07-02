import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor

from Wonder3D.mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from Wonder3D.mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from Wonder3D.mvdiffusion.data.single_image_dataset import SingleImageDataset

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

import warnings
warnings.filterwarnings("ignore")


def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    return ndarr


def sam_init():
    sam_checkpoint = os.path.join("Wonder3D", "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{0}")
    predictor = SamPredictor(sam)
    return predictor


def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def wonder3d_preprocess(predictor, input_image, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:,:,-1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len//2
        padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image.resize((320, 320), Image.Resampling.LANCZOS)


def load_wonder3d_pipeline(cfg):
    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=cfg.revision)
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=cfg.revision)
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs)
    unet.enable_xformers_memory_efficient_attention()

    # Move text_encode and vae to gpu and cast to torch.float16
    image_encoder.to(dtype=torch.float16)
    vae.to(dtype=torch.float16)
    unet.to(dtype=torch.float16)

    pipeline = MVDiffusionImagePipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, unet=unet, safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
        **cfg.pipe_kwargs
    )

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    # sys.main_lock = threading.Lock()
    return pipeline

def prepare_data(single_image, crop_size):
    dataset = SingleImageDataset(
        root_dir = None,
        num_views = 6,
        img_wh=[256, 256],
        bg_color='white',
        crop_size=crop_size,
        single_image=single_image
    )
    return dataset[0]


def run_wonder3d_pipeline(pipeline, cfg, single_image, guidance_scale, steps, seed, crop_size):

    batch = prepare_data(single_image, crop_size)

    pipeline.set_progress_bar_config(disable=True)
    seed = int(seed)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']]*2, dim=0).to(torch.float16)

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0).to(torch.float16)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(torch.float16)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(torch.float16)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
    # (B*Nv, Nce)
    # camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

    out = pipeline(
        imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale,
        num_inference_steps=steps,
        output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
    ).images

    bsz = out.shape[0] // 2
    normals_pred = out[:bsz]
    images_pred = out[bsz:]

    normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
    images_pred = [save_image(images_pred[i]) for i in range(bsz)]

    out = images_pred + normals_pred
    return *out, images_pred, normals_pred


#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min)
  x_max = min(W, x_max)
  y_min = max(0, y_min)
  y_max = min(H, y_max)
  w = x_max - x_min
  h = y_max - y_min
  image_witdh = W
  image_height = H

  bbox = [x_min, y_min, w, h, image_witdh, image_height]

  return bbox

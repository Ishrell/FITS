import argparse
from typing import List
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GRADIO_DEMO_DIR = os.path.join(REPO_ROOT, "gradio_demo")
if GRADIO_DEMO_DIR not in sys.path:
    sys.path.insert(0, GRADIO_DEMO_DIR)

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL, DDPMScheduler

from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModelRef
from src.unet_hacked_tryon import UNet2DConditionModel
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

from utils_mask import get_mask_location
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation


_CACHED_RUNTIME: dict[tuple[str, str], dict[str, object]] = {}


def _configure_torch_speedups() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _try_enable_xformers() -> bool:
    """Try to enable xformers for faster attention computation."""
    try:
        import xformers
        return True
    except ImportError:
        return False


def _enable_memory_optimizations(pipe) -> None:
    """Enable memory optimizations based on environment settings."""
    enable_attn_slicing = os.getenv("IDMVTON_ENABLE_ATTENTION_SLICING", "1").strip().lower() in {"1", "true", "yes"}
    enable_vae_tiling = os.getenv("IDMVTON_ENABLE_VAE_TILING", "1").strip().lower() in {"1", "true", "yes"}
    
    if enable_attn_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    
    if enable_vae_tiling:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image IDM-VTON inference for FITS")
    parser.add_argument("--person", required=True, help="Path to person image")
    parser.add_argument("--cloth", required=True, help="Path to cloth image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps (default: 20, recommend 15-25 for speed/quality balance)")
    parser.add_argument("--width", type=int, default=768, help="Output width (multiple of 64 recommended)")
    parser.add_argument("--height", type=int, default=1024, help="Output height (multiple of 64 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--garment_desc", default="upper body garment", help="Garment description prompt")
    return parser.parse_args()


def load_models(device: str, dtype: torch.dtype):
    base_path = "yisol/IDM-VTON"

    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", revision=None, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", revision=None, use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=dtype)
    unet_encoder = UNet2DConditionModelRef.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=dtype)

    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.unet_encoder = unet_encoder

    unet.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    pipe.to(device)
    pipe.unet_encoder.to(device)
    
    # Enable memory optimizations for faster inference on smaller GPUs
    _enable_memory_optimizations(pipe)

    return pipe


def get_cached_runtime(device: str, dtype: torch.dtype):
    cache_key = (device, str(dtype))
    runtime = _CACHED_RUNTIME.get(cache_key)
    if runtime is not None:
        return runtime

    _configure_torch_speedups()
    runtime = {
        "pipe": load_models(device, dtype),
        "parsing_model": Parsing(0),
        "openpose_model": OpenPose(0),
    }
    _CACHED_RUNTIME[cache_key] = runtime
    return runtime


def run_single_tryon(
    person_path: str,
    cloth_path: str,
    out_path: str,
    steps: int,
    seed: int,
    garment_desc: str,
    width: int,
    height: int,
) -> None:
    has_cuda = torch.cuda.is_available()
    device = "cuda:0" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32

    out_w = max(384, (int(width) // 64) * 64)
    out_h = max(512, (int(height) // 64) * 64)
    pose_w = max(256, out_w // 2)
    pose_h = max(384, out_h // 2)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    human_img = Image.open(person_path).convert("RGB").resize((out_w, out_h))
    garm_img = Image.open(cloth_path).convert("RGB").resize((out_w, out_h))

    runtime = get_cached_runtime(device, dtype)
    pipe = runtime["pipe"]
    parsing_model = runtime["parsing_model"]
    openpose_model = runtime["openpose_model"]

    keypoints = openpose_model(human_img.resize((pose_w, pose_h)))
    model_parse, _ = parsing_model(human_img.resize((pose_w, pose_h)))
    mask, _ = get_mask_location("hd", "upper_body", model_parse, keypoints)
    mask = mask.resize((out_w, out_h))

    _ = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)

    human_img_arg = _apply_exif_orientation(human_img.resize((pose_w, pose_h)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    model_device = "cuda" if has_cuda else "cpu"
    config_path = os.path.join(REPO_ROOT, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
    model_path = os.path.join(REPO_ROOT, "ckpt", "densepose", "model_final_162be9.pkl")
    args = apply_net.create_argument_parser().parse_args((
        "show",
        config_path,
        model_path,
        "dp_segm",
        "-v",
        "--opts",
        "MODEL.DEVICE",
        model_device,
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((out_w, out_h))

    with torch.inference_mode():
        if has_cuda:
            autocast_ctx = torch.cuda.amp.autocast()
        else:
            class _NoOp:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False
            autocast_ctx = _NoOp()

        with autocast_ctx:
            prompt = "model is wearing " + garment_desc
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            cloth_prompt = "a photo of " + garment_desc
            cloth_negative = "monochrome, lowres, bad anatomy, worst quality, low quality"
            cloth_prompt_list: List[str] = [cloth_prompt]
            cloth_negative_list: List[str] = [cloth_negative]

            (
                prompt_embeds_c,
                _,
                _,
                _,
            ) = pipe.encode_prompt(
                cloth_prompt_list,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=cloth_negative_list,
            )

            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, dtype)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, dtype)
            generator = torch.Generator(device=device).manual_seed(seed)

            images = pipe(
                prompt_embeds=prompt_embeds.to(device, dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                num_inference_steps=max(6, int(steps)),
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=out_h,
                width=out_w,
                ip_adapter_image=garm_img,
                guidance_scale=2.0,
            )[0]

    result = images[0]
    result.save(out_path)


def main() -> None:
    args = parse_args()
    run_single_tryon(
        person_path=args.person,
        cloth_path=args.cloth,
        out_path=args.output,
        steps=args.steps,
        seed=args.seed,
        garment_desc=args.garment_desc,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()

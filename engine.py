from typing import List, Optional, Union, Tuple
from torchvision.utils import save_image
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from torch import autocast
import inspect
import pdb
import numpy as np
import cv2
import torch
import torchvision.transforms as T

def inpaint(pipe, prompts, init_images, mask_images=None, control_images=None, strength=0.75, guidance_scale=7.5, generator=None, num_samples=1, n_iter=1):
    all_images = []
    transform = T.PILToTensor()
    for _ in range(n_iter):
        with autocast("cuda"):
            images = pipe(
                prompt=prompts,
                image=init_images,
                mask_image=mask_images,
                controlnet_conditioning_image=control_images, # NEW: Added ControlNet conditioning
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_samples,
            ).images
    for i in range(len(images)):
        all_images.append(transform(images[i]).unsqueeze(0))
    
    return torch.cat(all_images, dim=0)

def init_diffusion_engine(model_path, device, args):
    print('Initializing diffusion model: ', model_path)
    
    # NEW: Load the ControlNet model for depth
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth",
        torch_dtype=torch.float16,
    ).to(device)
    
    # NEW: Load the Stable Diffusion ControlNet Inpainting pipeline
    # The base model must be a regular SD model, not an inpainting one, because the ControlNet pipeline handles inpainting
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to(device)

    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    return pipe, generator

def generate(init_images, mask_images, pipe, generator, prompt=['lion'], device='cuda', depth_maps=None, strength=0.75, 
    guidance_scale=7.5, num_samples=1, n_iter=1):
    
    img_size = 512
    transform = T.Resize(img_size)
    init_images, mask_images = transform(init_images), transform(mask_images)
    
    # The depth maps must also be resized to match the image size
    depth_maps = transform(depth_maps)
    
    mask_images = mask_images[:,0,:,:].unsqueeze(1)
    
    # NEW: Pass the depth maps as the control_images argument
    results = inpaint(
        pipe, 
        [prompt]*mask_images.shape[0], 
        init_images, 
        mask_images, 
        control_images=depth_maps, 
        strength=strength, 
        guidance_scale=guidance_scale, 
        generator=generator, 
        num_samples=num_samples, 
        n_iter=n_iter
    )

    return results

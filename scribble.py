from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

image_location = "./images/scribble/final/original.png"

# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png")
image = load_image(image_location)

image = hed(image, scribble=True)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

prompt = "a chrysanthemum flower on its field"
negative_prompt = ""
resolution = 512
# prompt = "a handbag"
# negative_prompt = "low quality, blurry, unrealistic, out of proportion, nonsensical details, bad lighting, unrealistic textures, nonsensical composition"

image = pipe(prompt, image, num_inference_steps=20, width=resolution, height=resolution, negative_prompt=negative_prompt).images[0]

image.save('images/scribble/final/out_prompt1.png')

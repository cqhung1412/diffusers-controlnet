# https://huggingface.co/lllyasviel/sd-controlnet-hed
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/man.png")
image = load_image("./images/hed/final/original.jpg")


image = hed(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16
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

image = pipe("", image, num_inference_steps=20).images[0]

image.save('images/hed/final/out_noprompt.png')

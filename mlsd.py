# https://huggingface.co/lllyasviel/sd-controlnet-mlsd
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

image = load_image("./images/mlsd/final/original.png")

image = mlsd(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
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

# negative_prompt = "low quality, blurry, unrealistic"
image = pipe("a chinese living room", image, num_inference_steps=20, negative_prompt="").images[0]

image.save('images/mlsd/final/out_prompt2.png')

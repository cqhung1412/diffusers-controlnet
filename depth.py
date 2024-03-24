# https://huggingface.co/lllyasviel/sd-controlnet-depth
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image

depth_estimator = pipeline('depth-estimation')

# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
# image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
# image = load_image("./documentation-images/diffusers/autopipeline-inpaint.png")
image = load_image("./images/depth/final/original.jpg")

image = depth_estimator(image)['depth']
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# image = pipe("cute cat in Van Gogh style with starry sky", image, num_inference_steps=20).images[0]
# image = pipe("Batman in Van Gogh style with starry sky", image, num_inference_steps=20).images[0]
image = pipe("", image, num_inference_steps=20).images[0]

image.save('./images/depth/final/out_noprompt.png')

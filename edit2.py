import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda:1"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")

init_image = Image.open("/mnt/nas/swethamagesh/emotion/Asyrp_official/test_images/afhq/contents/flickr_dog_000772.png")
# init_image = init_image.resize((768, 512))
init_image.thumbnail((512, 512))

prompt = "a cartoon of a dog with a happy expression"
# 
# images = pipe(prompt=prompt, image=init_image).images
images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("/mnt/nas/swethamagesh/emotion/PiedPiper/editted_image.png")

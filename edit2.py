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

init_image = Image.open("/mnt/nas/swethamagesh/emotion/PiedPiper/iamge1.jpg")
init_image = init_image.resize((768, 512))

prompt = "Make the boy happier"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("/mnt/nas/swethamagesh/emotion/PiedPiper/editted_image.png")

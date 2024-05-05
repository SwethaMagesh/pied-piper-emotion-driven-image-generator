import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline
from transformers import pipeline

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda:1"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

init_image = Image.open("/mnt/nas/swethamagesh/emotion/PiedPiper/niks.jpg")
# init_image = init_image.resize((768, 512))
init_image.thumbnail((512, 512))

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
prompt = image_to_text("/mnt/nas/swethamagesh/emotion/PiedPiper/niks.jpg")
print(prompt[0]['generated_text'])

prompt = prompt[0]['generated_text']+" and very very angry"
# images = pipe(prompt=prompt, image=init_image).images
images = pipe(prompt=prompt, image=init_image, strength=0.5, guidance_scale=7.5).images
images[0].save("/mnt/nas/swethamagesh/emotion/PiedPiper/nikit.png")

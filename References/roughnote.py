from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

# Load pre-trained models - Replace these with your desired models
text_encoder = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

def edit_image_with_text(image, caption):
  """
  Edits an existing image based on a text caption using CLIP and Stable Diffusion.

  Args:
      image: Path to the existing image.
      caption: Text caption describing the desired edit.

  Returns:
      Edited image as a PIL Image object.
  """
  # Encode caption and image using CLIP
  input_ids = text_encoder(caption, return_tensors="pt")
  with torch.no_grad():
      image_features = image_encoder.get_image_features(image)

  # Generate initial noisy image
  latent_diffusion = model.scheduler.config.latent_diffusion
  init_image = latent_diffusion.sample(batch_size=1)

  # Iteratively refine the image based on CLIP feedback
  for i in range(100):
    # Generate text embeddings for the caption again
    text_features = image_encoder(input_ids.clone())

    # Get CLIP similarity score between current image and caption
    clip_loss = model.compute_loss(text_features=text_features, latent=init_image)["loss"]

    # Use the CLIP loss to guide the diffusion process
    init_image = model.diffuser.denoise(init_image, t=i, clip_loss=clip_loss)

  # Decode the final latent image into a PIL image
  edited_image = model.image_decoder(init_image)
  return edited_image

# Example usage
image_path = "image1.jpg"
caption = "Make the boy more happy"

edited_image = edit_image_with_text(image_path, caption)

# Save or display the edited image
edited_image.save("edited_image.jpg")

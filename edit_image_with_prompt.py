from PIL import Image

# Import necessary modules
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from diffusers import StableDiffusionPipeline
import torch

# Function to predict text description of an image
def predict(image, device="cuda:1"):
    """
    Predict the text description of an image.
    
    Args:
        image (PIL.Image): Image to be described
        device (str): Device to run the model on (default is "cuda:1")
    
    Returns:
        list of str: Predicted text description of the image
    """
    # Load pre-trained model and tokenizer
    loc = "ydshieh/vit-gpt2-coco-en"
    feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
    tokenizer = AutoTokenizer.from_pretrained(loc)
    model = VisionEncoderDecoderModel.from_pretrained(loc)

    # Move model to the specified device
    model = model.to(device)
    # Extract pixel values from the image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generate text description for the image
    with torch.no_grad():
        model.to('cpu')
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    # Decode the output text
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds

# Load the initial image
init_image = Image.open('/mnt/nas/swethamagesh/emotion/Asyrp_official/custom/test/sad1.jpg')
init_image = init_image.resize((768, 512))
init_image.thumbnail((256, 256))

# Predict text description of the image
with Image.open('/mnt/nas/swethamagesh/emotion/Asyrp_official/custom/test/sad1.jpg') as image:
    preds = predict(image)

print(preds)

# Construct prompt for image transformation
prompt = str(preds[0]) + " and make the picture very very happy"

# Initialize StableDiffusionPipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None,
                                               requires_safety_checker=False)
pipe = pipe.to("cuda")

print(prompt)

# Transform the image based on the prompt
images = pipe(prompt, image=init_image, strength=0.1, guidance_scale=2).images
for i in range(len(images)):
    images[i].save(f"image_{i}.png")

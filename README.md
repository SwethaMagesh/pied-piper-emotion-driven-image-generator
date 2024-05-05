# pied-piper-emotion-driven-image-generator
## Things tried out
1. Textual inversion
2. Simple pipeline based on hugging face
3. Basic StyleGAN
4. VAE based faceflex
5. Stability AI API
## References
1. ASYRP - paper 
2. Emogen - not useful for image editing (only generation)
3. Textual inversion
- colab file (starter) 
- check lora nd dreambooth code
4. StyleGAN
 - images generated were giving good results for familiar data, new images suffer from distortions like(closed eyes, background mixing, jumbled facial features)
 github link: 
 (https://github.com/IIGROUP/TediGAN)
5. Face-flex
  - This GitHub project employs Variational Autoencoders (VAEs) to modify facial expressions
  - Emphasis on adding specific emotions to images
  - Identified distorted outputs images
  - Project predominantly generates smiling emotions, overlooking others
 

-----
 ## Notes
- Metrics:
- Inception score & FID (Frechet Inception dis)
- Use a emotion classifier to find our generated image accuracy
-   Train an additional image classifier specifically for emotions (happy, sad, etc.). Use the Inception Score concept, but feed the generated images into both the InceptionV3 network and the emotion classifier. A good model should achieve high Inception Score while also assigning the correct emotion label with high confidence in the emotion classifier.


###


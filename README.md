# pied-piper-emotion-driven-image-generator

## References
1. ASYRP 
- paper to be gone through
2. Emogen 
- not useful for image editing (only generation)
3. Textual inversion
- colab file (starter) not useful
- check lora nd dreambooth code
4.StyleGAN
 - images generated were giving good results for familiar data, new images suffer from distortions like(closed eyes, background mixing, jumbled facial features)
 github link: 
 (https://github.com/IIGROUP/TediGAN)

-----
 ## Notes
- Metrics:
- Inception score & FID (Frechet Inception dis)
- Use a emotion classifier to find our generated image accuracy
-   Train an additional image classifier specifically for emotions (happy, sad, etc.). Use the Inception Score concept, but feed the generated images into both the InceptionV3 network and the emotion classifier. A good model should achieve high Inception Score while also assigning the correct emotion label with high confidence in the emotion classifier.

## Other ideas
###


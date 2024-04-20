# pied-piper-emotion-driven-image-generator

## References
1. ASYRP 
- paper to be gone through
2. Emogen 
- not useful for image editing (only generation)
3. Textual inversion
- colab file (starter) not useful
- check lora nd dreambooth code

-----
 ## Notes
- Metrics:
- Inception score & FID (Frechet Inception dis)
- Use a emotion classifier to find our generated image accuracy
-   Train an additional image classifier specifically for emotions (happy, sad, etc.). Use the Inception Score concept, but feed the generated images into both the InceptionV3 network and the emotion classifier. A good model should achieve high Inception Score while also assigning the correct emotion label with high confidence in the emotion classifier.

## Other ideas
###

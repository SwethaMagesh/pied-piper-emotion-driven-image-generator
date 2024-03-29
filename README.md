﻿# pied-piper-emotion-driven-image-generator

 ## Notes
- Metrics:
- Inception score & FID (Frechet Inception dis)
- Use a emotion classifier to find our generated image accuracy
-   Train an additional image classifier specifically for emotions (happy, sad, etc.). Use the Inception Score concept, but feed the generated images into both the InceptionV3 network and the emotion classifier. A good model should achieve high Inception Score while also assigning the correct emotion label with high confidence in the emotion classifier.

## Other ideas
### can we condition noise with emotion labels in diffusion?
###

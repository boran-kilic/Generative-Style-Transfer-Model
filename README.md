# Generative-Style-Transfer-Model
In this project, a generative style transfer model for unpaired image-to-image translation is implemented. Among different Generative Adversarial Networks (GANs), the Cycle Consistent Adversarial Networks (CycleGAN) algorithm is used for transforming random real world photos to different art styles. Three styles are chosen among the available style datasets, these are Van Gogh and Ukiyo-e painting as well as a general engraving dataset. The images are generated by training the CycleGAN with the content image training dataset and the style datsets of each style. The generated images are then evaluated by different metrics; firstly by ArtFID followed by ArtDBF, our generated metric for this project.

CycleGAN.ipynb contains the training and image generation codes
preprocess.py is the preprocessing code for the training dataset
ArtFID_ArtDBF folder contains the evaluation metric codes

# Image Generation and Classification Project with GAN, Stable Diffusion and VGG16
This project implements a system to generate images using a GAN and Stable Diffusion, comparing them with real images taken from COCO dataset.
To predict wheter an image is real or generated, we use a VGG16,of which we have removed the last layer and replaced it with an SVM, wich we have re-trained to allow us to predict the class (real or generated), of the images

It also includes the metrics as accuracy, precision, recall and F1-score.




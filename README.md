# Image Generation and Classification Project with GAN, Stable Diffusion and VGG16
This project implements a system to generate general images using a GAN and Stable Diffusion, comparing them with real images taken from COCO dataset.
To predict wheter an image is real or generated, we use a VGG16,of which we have removed the last layer and replaced it with an SVM, wich we have re-trained to allow us to predict the class (real or generated), of the images.
## Stable Diffusion
Stable diffusion is a pre-trained model "stabilityai/stable-diffusion-2-1" that takes annotations from the coco dataset and generates an image with them.
This is an example of image generated by Stable diffusion:

## GAN
The GAN model, on the other hand, was created from scratch by us, has 5 convolutional layers, has a batch size of 128, generates the output images with a resolution of 64x64 and it was trained on coco images over 100 epochs.
these are the result of 1st epoch and 100th epoch:



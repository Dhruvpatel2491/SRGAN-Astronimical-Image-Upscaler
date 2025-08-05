# Super-Resolution GAN for Astronomical Imaging

## Project Overview

This project implements a Super-Resolution Generative Adversarial Network (SRGAN) to enhance the resolution of astronomical images. By using a GAN architecture, the model not only upscales low-resolution images but also synthesizes realistic, high-frequency details that are often lost during traditional upsampling methods. This is particularly valuable for astronomical data, where fine details are critical for scientific analysis.

## Key Features

* **GAN Architecture:** Implements a full Generative Adversarial Network with a Generator and a Discriminator.
* **Bug Fixes:** Corrects and refines the training loop and model definitions to ensure proper GAN behavior.
* **Astronomical Data Tailored:** The model architecture and preprocessing steps are configured to handle single-channel grayscale images, which are common in astronomical datasets.
* **Custom Training Loop:** A custom TensorFlow training loop is implemented for effective adversarial training.
* **Perceptual Loss:** The generator's loss function combines both adversarial loss and content loss (Mean Squared Error), encouraging the model to produce both realistic and structurally accurate images.
* **Quantitative Evaluation:** The project includes functionality to measure and visualize image quality using industry-standard metrics:
    * **Peak Signal-to-Noise Ratio (PSNR):** A metric that measures the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher PSNR indicates better quality.
    * **Structural Similarity Index (SSIM):** A perception-based model that measures the similarity between two images. It is often considered a more accurate measure of perceived quality than PSNR.
* **Image Visualization:** Provides code to plot and compare low-resolution, super-resolved, and high-resolution ground truth images, offering a clear visual assessment of the model's performance.

## GAN Architecture
<br>
<p align="center">
  <img src="https://www.labellerr.com/blog/what-is-gan-how-does-it-work/" alt="GAN Architecture Diagram" width="500">
  <br>
  <em>A high-level diagram illustrating the adversarial training process of a GAN.</em>
</p>

### Generator
The generator is a deep convolutional neural network responsible for upscaling the low-resolution image. Its architecture consists of:
* A series of residual blocks to learn a complex mapping from low to high resolution.
* Upsampling layers using `tf.nn.depth_to_space` to increase the spatial dimensions of the image.

### Discriminator
The discriminator acts as a critic, tasked with distinguishing between real high-resolution images and fake images produced by the generator. Its architecture is a standard convolutional classifier:
* A series of convolutional layers with LeakyReLU activations.
* A final dense layer with a sigmoid activation to output a probability score (real or fake).

## Getting Started

### Prerequisites
* Python 3.x
* TensorFlow
* OpenCV
* Scikit-image
* Seaborn

### Installation
Clone this repository and install the required packages:

```bash
git clone [https://github.com/Dhruvpatel2491/SRGAN-Astronimical-Image-Upscaler.git](https://github.com/Dhruvpatel2491/SRGAN-Astronimical-Image-Upscaler.git)
cd SRGAN-Astronimical-Image-Upscaler
pip install -r requirements.txt
# Automatic Differentiation, Filtering, and Convolutional Networks

This repository contains my implementation of key computer vision and deep learning techniques. The project is divided into three main parts:

1. **Image Filtering and Convolutions**: 
   - Implemented convolution operations from scratch and applied various image filtering techniques (such as Gaussian smoothing).
   - Explored different padding methods (zero-padding and symmetric-padding) for edge detection on images.
   - Visualized results for edge detection filters using 1D and 2D kernels on the provided zebra and cameraman images.

2. **Forward-mode and Backward-mode Automatic Differentiation**: 
   - Implemented both **forward-mode** and **reverse-mode** automatic differentiation on scalar functions, handling basic arithmetic, trigonometric, and exponential operations.
   - Explored how the **chain rule** can be applied in automatic differentiation by calculating partial derivatives of composite functions.

3. **Convolutional Neural Networks (CNNs)**:
   - Built a CNN from scratch, implementing layers for convolutions, batch normalization, and ReLU activation functions.
   - Utilized forward and backward propagation techniques to compute gradients and optimize model parameters.
   - The CNN was tested on image classification tasks using sample datasets and custom layers.

## Project Structure

- `part1_image_filtering`: Contains code for image filtering, convolution, and edge detection.
- `part2_autodiff`: Implementation of forward-mode and backward-mode automatic differentiation for scalar variables.
- `part3_cnn`: The CNN implementation with forward and backward propagation. Includes convolution, batch normalization, ReLU layers, and testing on sample datasets.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/manisha-goyal/cnn-image-filtering-and-differentiation.git
   cd cnn-image-filtering-and-differentiation
   ```

2. Install necessary libraries:
   ```bash
   pip install numpy matplotlib torch
   ```

3. Each part of the project can be run individually in the respective Jupyter notebooks or Python scripts.

## File Descriptions

- **part1_image_filtering**:
  - `image_filtering.ipynb`: Jupyter notebook that contains code for applying various filters, convolutions, and visualizing the output.
  
- **part2_autodiff**:
  - `auto_diff.ipynb`: Jupyter notebook containing forward-mode and backward-mode automatic differentiation.
  
- **part3_cnn**:
  - `convolutional_networks.py`: Python script where the CNN layers are implemented.
  - `convolutional_networks.ipynb`: Jupyter notebook that runs tests for the CNN implementation.

## Key Learnings
- Understanding and implementing convolution operations from scratch helped reinforce the concept of filtering in computer vision.
- The forward and reverse mode automatic differentiation deepened my understanding of how gradients are computed and accumulated in deep learning models.
- Building a CNN from scratch and implementing batch normalization and activation layers provided a strong foundation for understanding backpropagation and gradient descent.

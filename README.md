# Convolutional Neural Networks

**Author:** Miguel Carrasco Belmar  
**Institution:** Boise State University  
**Course:** Data Science Programming Assignment  

---

## Abstract

This project introduces the structure and function of **Convolutional Neural Networks (CNNs)**, a class of deep learning architectures that have become fundamental in modern computer vision.  
The notebook accompanying this report provides a complete demonstration of CNN principles — beginning with the visualization of convolution, activation, and pooling operations on a sample image — and culminates with the implementation and training of a CNN model on the **MNIST handwritten digits dataset**.  
Through theory, experimentation, and quantitative evaluation, this project illustrates how CNNs transform raw pixel data into hierarchical feature representations and achieve high classification accuracy on real-world visual tasks.

---

## 1. Introduction

Artificial Neural Networks (ANNs) are computational systems inspired by the human brain’s structure, where interconnected neurons process data collectively to recognize patterns.  
A **Convolutional Neural Network (CNN)** is a specialized form of ANN designed to handle **spatial data**, such as images, where local patterns and pixel arrangements carry meaning.

CNNs are now used extensively in:
- **Facial recognition and biometrics**
- **Autonomous vehicle perception**
- **Medical image analysis**
- **Handwriting and character recognition**

The goal of this project is twofold:
1. To illustrate, in a visual and computational way, how CNN layers (convolution, activation, and pooling) extract hierarchical features from images.  
2. To implement and evaluate a complete CNN model capable of classifying handwritten digits using the MNIST dataset with high accuracy.

---

## 2. Background: Neural Networks

A neural network learns by adjusting the strength of connections (weights) between its neurons in order to minimize prediction error.  
Each neuron performs a linear combination of its inputs followed by a **non-linear activation function**, which enables the network to approximate complex mappings.

A typical feed-forward neural network consists of:
- **Input Layer:** Accepts raw data (e.g., image pixels or numeric features).  
- **Hidden Layers:** Perform transformations to extract intermediate patterns.  
- **Output Layer:** Produces final predictions (e.g., categorical labels).

However, dense networks ignore spatial relationships between pixels, which makes them inefficient for image analysis.  
CNNs address this limitation through **local connectivity**, **weight sharing**, and **hierarchical feature extraction**.

---

## 3. Convolutional Neural Networks: Structure and Function

CNNs are designed to automatically identify spatial hierarchies in image data.  
They process local image patches using small learnable filters, then progressively combine these local features into complex global structures.

The architecture consists of the following key components:

1. **Input Layer**  
   Receives image data as a 3D tensor (height × width × channels).  
   No computation occurs at this stage; the pixel values are simply passed to subsequent layers.

2. **Convolutional Layer**  
   The core component of the network.  
   Small filters (kernels) convolve across the input image to detect local features such as edges or corners.  
   Each kernel generates a feature map that records the spatial locations where that feature appears.  
   As more convolutional layers are added, the network learns increasingly abstract features — progressing from edges to textures, shapes, and object parts.

3. **Activation Layer**  
   Introduces non-linearity to the model, enabling it to learn complex relationships.  
   The **Rectified Linear Unit (ReLU)** is used in this project, defined as  
   \[
   f(x) = \max(0, x)
   \]  
   It accelerates convergence and mitigates vanishing gradients during training.

4. **Pooling Layer**  
   Reduces the spatial dimensions of feature maps while preserving the most informative activations.  
   **Max Pooling** is employed to select the maximum value within a region (e.g., 2×2).  
   This improves computational efficiency and provides translational invariance.

5. **Fully Connected (Dense) Layers**  
   After several convolution and pooling stages, the resulting features are flattened and passed through dense layers.  
   These layers integrate the learned representations to perform classification.

6. **Output Layer**  
   The final layer uses the **Softmax** activation function to produce a probability distribution across classes.  
   For MNIST, this corresponds to ten categories representing digits 0–9.

---

## 4. Layer Demonstration: Convolution, Activation, and Pooling

To visualize how CNN operations transform image data, a single grayscale image (“Ganesh.jpg”) was processed using TensorFlow.  
The following sequence of operations was performed:

1. **Convolution:**  
   A 3×3 edge-detection kernel was applied, producing a feature map highlighting intensity changes.  
   Bright regions corresponded to strong edges and contours.

2. **Activation (ReLU):**  
   Negative values in the convolved image were replaced with zeros, retaining only strong positive responses.

3. **Pooling (Max Pooling):**  
   The activated feature map was downsampled to reduce size while preserving dominant visual features.

The resulting progression — from raw image → convolution → activation → pooling — visually demonstrates how CNNs extract and condense visual information.  
Edges and outlines become clearer, and redundant pixel details are suppressed.

---

## 5. CNN Implementation on MNIST Dataset

The **MNIST dataset** consists of 70,000 grayscale images (60,000 for training and 10,000 for testing), each depicting a handwritten digit from 0 to 9 at 28×28 pixels resolution.  
The dataset serves as a standard benchmark for image classification algorithms.

### 5.1 Data Preparation
- The data was loaded via `tensorflow.keras.datasets.mnist`.  
- Input images were normalized to the [0,1] range.  
- Data was reshaped into 4-D tensors `(samples, height, width, channels)`.  
- Labels were one-hot encoded for multi-class classification.

### 5.2 Model Architecture

The following architecture was implemented using Keras:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

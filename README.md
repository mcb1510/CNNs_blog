# 🧠 Introduction to Convolutional Neural Networks (CNNs)
**Author:** Miguel Carrasco Belmar  
**Course:** Data Science Programming Assignment  
**Institution:** Boise State University  

---

## 📘 Overview

Convolutional Neural Networks (CNNs) are a powerful class of deep learning architectures that have revolutionized the field of **computer vision**.  
Unlike traditional machine learning models, which rely heavily on handcrafted features, CNNs **learn directly from data** by automatically detecting and combining hierarchical visual patterns.

This project introduces CNNs through both **conceptual explanation** and **practical experimentation**.  
It combines a detailed blog-style discussion of how CNNs work with a full **Python implementation** using **TensorFlow/Keras**, applied to the **MNIST handwritten digits dataset**.  
The notebook includes detailed visualizations showing how each layer transforms images, from raw pixels to feature maps to predictions.

---

## 🧩 Why This Project Matters

The ability of CNNs to understand images without manual feature design is what powers modern AI technologies, including:

- 👁️ **Facial recognition** systems (Face ID, surveillance)  
- 🚗 **Autonomous driving** (object detection, lane tracking)  
- 🏥 **Medical imaging** (tumor detection, X-ray classification)  
- 📱 **Smartphones and social media** (photo tagging, filters)  
- 🛰️ **Remote sensing** (satellite and aerial image analysis)  

Understanding CNNs is therefore fundamental to becoming an effective data scientist or AI researcher.  
This project demonstrates both *how* CNNs function and *why* they are so effective in analyzing visual data.

---

## 🧠 Historical Background

Before CNNs, image recognition tasks relied on manual feature extraction methods such as **SIFT (Scale-Invariant Feature Transform)** and **HOG (Histogram of Oriented Gradients)**.  
While effective in some cases, these methods required human expertise and could not generalize well.

In 1998, **Yann LeCun** introduced **LeNet-5**, one of the first convolutional neural networks, to recognize handwritten digits.  
Two decades later, CNNs like **AlexNet (2012)**, **VGGNet (2014)**, and **ResNet (2015)** achieved breakthroughs on large-scale datasets such as **ImageNet**, marking the beginning of the deep learning era.

This project revisits that legacy by implementing a modernized version of LeNet for the MNIST dataset.

---

## 🔢 Mathematical Foundations

### 1. Convolution Operation

At the core of CNNs lies the **convolution operation**, which captures local dependencies in data.  
For an image \( I \) and a kernel \( K \):

\[
S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(m, n) \cdot K(i - m, j - n)
\]

This operation slides a small filter \( K \) across the image \( I \), producing a **feature map** \( S \).  
Each filter extracts a specific pattern — for example, a vertical edge or a diagonal line.  
During training, CNNs learn optimal filter values that best represent meaningful patterns in the data.

---

### 2. Activation Functions

After convolution, CNNs apply a **non-linear activation function** to model complex relationships.  
The most common is the **Rectified Linear Unit (ReLU)**:

\[
f(x) = \max(0, x)
\]

This operation removes negative activations and keeps only the strongest responses, helping the network converge faster and avoid the vanishing gradient problem.

---

### 3. Pooling Operation

Pooling reduces the dimensionality of feature maps while preserving the most important information.  
**Max pooling** takes the maximum value within each window:

\[
P(i, j) = \max_{(m, n) \in R(i, j)} S(m, n)
\]

This process introduces **spatial invariance**, meaning that slight shifts or rotations in the image won’t affect recognition.

---

### 4. Fully Connected Layers and Softmax

After multiple rounds of convolution and pooling, the feature maps are flattened into a vector and fed into **fully connected layers**, which act as a traditional neural network.  
The final layer uses the **Softmax** function to output probabilities:

\[
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\]

Where \( K \) is the number of classes (in MNIST, 10 digits).  
This ensures all outputs sum to 1, representing class probabilities.

---

## 🧩 Layer-by-Layer Walkthrough

A CNN processes images hierarchically.  
Below is the typical structure:

| Layer | Function | Description |
|--------|-----------|-------------|
| **Input Layer** | Accepts image input | Usually of shape (height, width, channels) |
| **Convolutional Layer** | Detects local patterns | Applies filters that identify edges, textures, or shapes |
| **Activation Layer (ReLU)** | Adds non-linearity | Ensures network can model complex functions |
| **Pooling Layer (MaxPooling)** | Reduces size | Summarizes strong activations and prevents overfitting |
| **Flatten Layer** | Converts 2D to 1D | Prepares data for fully connected layers |
| **Dense (Fully Connected) Layer** | Combines features | Learns global associations for classification |
| **Output Layer (Softmax)** | Predicts classes | Produces final probabilities for each category |

---

## 🧮 Example: Visualizing CNN Operations on an Image

Before training on MNIST, it is instructive to visualize how convolution, activation, and pooling affect an image.  
We use a simple **edge detection kernel** to illustrate how CNNs extract visual features.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

image = tf.io.read_file('Ganesh.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [3, 3, 1, 1])

conv = tf.nn.conv2d(image, kernel, strides=1, padding='SAME')
relu = tf.nn.relu(conv)
pooled = tf.nn.pool(relu, window_shape=(2, 2), pooling_type='MAX', strides=(2, 2), padding='SAME')

plt.figure(figsize=(15, 5))
for i, stage in enumerate([conv, relu, pooled]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(tf.squeeze(stage))
    plt.axis('off')
    plt.title(['Convolution', 'Activation (ReLU)', 'Pooling'][i])
plt.show()

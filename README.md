# Convolutional Neural Networks

**Author:** Miguel Carrasco Belmar  
Have you ever counted how many times you pick up your phone each day? It’s estimated that an average person unlocks their phone 110 to 150 times per day. Each time you unlock your phone using Face ID or facial recognition, you’re actually using a Convolutional Neural Network, over 100 times a day! Convolutional Neural Networks (CNNs) are a type of neural network widely used in Artificial Intelligence (AI), especially in computer vision tasks such as image classification, object detection, and image segmentation.
<p align="center">
  <img src="/images/banner.png" alt="CNN feature banner">
</p>

## Background: What are Neural Networks
A Neural Network is a computational model within the domain of Deep Learning, a subfield of Machine Learning.
It is composed of interconnected processing units, known as artificial neurons, organized in multiple layers that collectively learn to approximate complex functions.
This architecture is inspired by the structure and functioning of the human brain. Each neuron receives input, processes it through a mathematical function (typically involving weighted summations and activation functions), and transmits the output to subsequent neurons.
Through an iterative training process, the network learns to adjust its internal parameters (weights and biases) to minimize prediction error.
Once trained, the network can recognize patterns in data and make predictions in diverse categories such as image recognition, speech processing, and natural language understanding.

## How do Neural Networks work?
Artificial neural networks are designed to emulate the way biological neurons transmit and process signals.
They are composed of distinct layers, each performing a specific function in the learning process:
<p align="center">
  <img src="/images/layers.png" alt="layers" width="600">
</p>

**Input Layer:**
This layer receives raw data from the external environment. Each input node corresponds to one feature of the dataset (for example, pixel intensity in an image). The input layer transmits this information to subsequent layers without modification.

**Hidden Layers:**
These layers perform the majority of the computation. They transform the input data into increasingly abstract representations through weighted connections and activation functions. A deep neural network may contain numerous hidden layers, each responsible for capturing progressively higher-level features.

**Output Layer:**
The final layer produces the network’s output, such as a class label or probability distribution.
In binary classification tasks, this layer may consist of a single neuron that outputs a value between 0 and 1.
In contrast, multi-class classification problems require multiple output neurons, each representing a distinct class.

Depending on how data propagates through the layers, neural networks can assume various architectures. Among these, Convolutional Neural Networks are particularly effective for processing visual and spatial data.

## What Are Convolutional Neural Networks (CNNs)
Convolutional Neural Networks (CNNs) are a specialized class of neural networks primarily designed to process and extract meaningful features from grid-like data structures, such as images or videos. In visual data, spatial patterns, for example, edges, textures, and shapes, play a crucial role in understanding the content. CNNs are particularly effective in capturing these spatial hierarchies, which makes them the foundation of most computer vision applications today.

<p align="center">
  <img src="/images/cnns.PNG" alt="CNN feature banner">
</p>

A CNN is formed by multiple layers that work together to transform raw pixel data into abstract, high-level representations. The key layers that define the architecture of a CNN are:

**1.- Input Layer:** This layer receives the raw image data, typically represented as a three-dimensional matrix of pixel intensity values corresponding to the image’s height, width, and color channels (RGB). The input layer does not perform any computation; it simply feeds the pixel values to the next layer.

**2.- Convolutional Layers:** This is the core building block of a CNN. It applies small, learnable filters (called kernels) that slide over the input image to detect localized features such as edges, corners, or gradients. Each filter produces a feature map that highlights where specific patterns occur in the image. By stacking multiple convolutional layers, the network learns hierarchical featuresfrom simple edges in early layers to more complex structures or objects in deeper ones.


**3.- Activation Layers:** After each convolution operation, an activation function introduces non-linearity into the model, enabling it to learn complex patterns. The most common activation function is the Rectified Linear Unit (ReLU), which replaces all negative values with zero. This helps the CNN converge faster (i.e., find optimal parameters efficiently) and prevents the vanishing gradient problem, a common issue where gradients become too small for effective learning in deep networks.


**4.- Pooling Layers:** They reduce the spatial size of the feature maps while retaining the most significant information. The most widely used pooling method is Max Pooling, which selects the maximum value from each local region (for example, a 2×2 window). This process reduces computational cost and makes the CNN more robust to variations and distortions in the input image.


**5.- Fully Connected Layers:** After several convolutional and pooling layers, the extracted features are flattened into a one-dimensional vector and passed through one or more fully connected (dense) layers. These layers combine the learned features to make high-level inferences and form the basis for the final classification.


**6.- Output Layer:** This is the final layer of the CNN that produces the prediction. For instance, in a multi-class classification task, the Softmax activation function is commonly used to convert the output into a probability distribution across all possible categories (e.g., cat = 0.85, dog = 0.10, car = 0.05).

## Experimental Implementation: CNN Model for Handwritten Digit Recognition Using the MNIST Dataset
To demonstrate how Convolutional Neural Networks operate in practice, we implemented a simple CNN model using the MNIST handwritten digits dataset.

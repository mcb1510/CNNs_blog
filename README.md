# Convolutional Neural Networks

**Author:** Miguel Carrasco Belmar  
<br>
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
  <img src="/images/cnns.PNG" alt="cnns">
</p>

A CNN is formed by multiple layers that work together to transform raw pixel data into abstract, high-level representations. The key layers that define the architecture of a CNN are:

**1.- Input Layer:** This layer receives the raw image data, typically represented as a three-dimensional matrix of pixel intensity values corresponding to the image’s height, width, and color channels (RGB). The input layer does not perform any computation; it simply feeds the pixel values to the next layer.

**2.- Convolutional Layers:** This is the core building block of a CNN. It applies small, learnable filters (called kernels) that slide over the input image to detect localized features such as edges, corners, or gradients. Each filter produces a feature map that highlights where specific patterns occur in the image. By stacking multiple convolutional layers, the network learns hierarchical features from simple edges in early layers to more complex structures or objects in deeper ones.


**3.- Activation Layers:** After each convolution operation, an activation function introduces non-linearity into the model, enabling it to learn complex patterns. The most common activation function is the Rectified Linear Unit (ReLU), which replaces all negative values with zero. This helps the CNN converge faster (i.e., find optimal parameters efficiently) and prevents the vanishing gradient problem, a common issue where gradients become too small for effective learning in deep networks.


**4.- Pooling Layers:** They reduce the spatial size of the feature maps while retaining the most significant information. The most widely used pooling method is Max Pooling, which selects the maximum value from each local region (for example, a 2×2 window). This process reduces computational cost and makes the CNN more robust to variations and distortions in the input image.


**5.- Fully Connected Layers:** After several convolutional and pooling layers, the extracted features are flattened into a one-dimensional vector and passed through one or more fully connected (dense) layers. These layers combine the learned features to make high-level inferences and form the basis for the final classification.


**6.- Output Layer:** This is the final layer of the CNN that produces the prediction. For instance, in a multi-class classification task, the Softmax activation function is commonly used to convert the output into a probability distribution across all possible categories (e.g., cat = 0.85, dog = 0.10, car = 0.05).

The following image illustrates how the CNN layers transform the input step by step:

<p align="center">
  <img src="/images/elephant.png" alt="cnn_example">
</p>

**1.- Original Image:**
We start with a colorful input image of an elephant (716 × 788 pixels).

**2.- Grayscale & Resize:**
The image is simplified by removing color and resizing while keeping important structure.

**3.- Convolution:**
A filter scans the image and highlights edges and shapes, making contours more visible.

**4.- Activation (ReLU):**
Negative values are removed, keeping only strong feature responses.

**5.- Pooling:**
The feature map is downsampled, reducing size while retaining the most important information.

This sequence shows how CNNs gradually extract and condense visual features before classification.


## Experimental Implementation: CNN Model for Handwritten Digit Recognition Using the MNIST Dataset
To demonstrate how Convolutional Neural Networks operate in practice, we implemented a simple CNN model using the MNIST handwritten digits dataset. 
We used Keras, a high-level deep learning library in Python (via `tensorflow.keras`), to build and train the model.  
The goal of this experiment is to recognize handwritten digits automatically from image data.


### **MNIST**
The dataset, MNIST, consists of  70,000 grayscale images of handwritten digits, with 60,000 for training and 10,000 for testing, each 28 × 28 pixels.
Reshaping, the images were reshaped to 4D tensors with shape (samples, 28, 28, 1) to fit the Keras CNN layers, and normalized from [0, 255] to [0, 1], which helps gradients behave better during training. Labels were converted to one-hot encoding so the network can output a probability for each digit. Here is a visualization of the training data set
<p align="center">
  <img src="/images/dataset_preview.PNG" alt="dataset_preview">
</p>

### **CNN Architecture**
Then we built our Convolutional Neural Network. For a small project like this, we can achieve good results with a compact model with the following structure: 

<br>
**Two convolution–pooling blocks**
The first block uses a 3×3 convolution with 32 filters followed by 2×2 max pooling, and the second block uses a 3×3 convolution with 64 filters followed by another 2×2 max pooling.
Together, these layers detect simple local patterns (edges, strokes, and corners) in the early layers and more complex combinations of strokes (loops and intersections) in the deeper layers.

**Flatten and dense layer**
The resulting feature maps are flattened into a 1D vector and passed to a dense layer with 128 neurons and ReLU activation. This layer combines the learned features into a compact representation useful for classification.

**Dropout for regularization**
A dropout layer with a rate of 0.5 randomly “turns off” half of the neurons during training. This helps prevent overfitting (model memorizing the training data) by forcing the network to rely on multiple features rather than memorizing specific patterns.

**Output layer with Softmax**
The final dense layer has 10 neurons with Softmax activation, producing a probability distribution over the 10 digit classes (0–9).
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN architecture
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

```

### **Training Setup**

The model was trained using the following setup:

**Optimizer:** Adam, which adapts the learning rate during training.

**Loss function:** Categorical cross-entropy, appropriate for multi-class classification with one-hot labels.

**Metrics:** Accuracy on both the training and validation sets.

**Training configuration:** 5 epochs, batch size of 128, with 10% of the training data held out as a validation set.

After training, we calculated the model's accuracy and loss and displayed it
<p align="center">
  <img src="/images/curves.PNG" alt="curves">
</p>

And we tested our model
```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
```
We obtained an excellent score of 99.07% Test Accuracy.
To better understand which digits the model recognizes well and where it struggles, we computed a confusion matrix on the test set.

In this matrix, each row corresponds to the true digit. Each column corresponds to the predicted digit. Values along the main diagonal represent correct classifications, while off-diagonal values represent mistakes.
<p align="center">
  <img src="/images/confusion_matrix.PNG" alt="conf_matrix">
</p>
In total, the model correctly predicts 9,907 out of 10,000 test images, which corresponds to our 99.07% test accuracy. This confirms that the CNN has learned highly effective feature representations for handwritten digit recognition. Here is a picture of the prediction output from our model
<p align="center">
  <img src="/images/output.PNG" alt="conf_matrix">
</p>

### Conclusion

This project showed how Convolutional Neural Networks turn raw pixels into useful predictions, starting from simple operations like convolution, ReLU, and pooling, and ending with a full model that can recognize handwritten digits. After preprocessing the data and building a small but effective architecture, the model reached high accuracy on unseen test images, proving that these learned features are truly useful, not just memorized patterns. At the same time, the experiment highlighted that MNIST is a relatively simple problem, and that more complex, real world tasks would require deeper models, more data, and additional techniques. Overall, the project not only explains what CNNs are, it also demonstrates how data preprocessing, model design, and visualization come together in a real data science workflow, from understanding the method to evaluating its strengths and limitations.

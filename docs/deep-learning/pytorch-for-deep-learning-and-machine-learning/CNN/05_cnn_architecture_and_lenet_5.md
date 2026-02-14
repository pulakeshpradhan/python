---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/02_CNN/05_CNN_Architecture_and_LeNet_5.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="TaIw9nxUpipf" -->
# **CNN Architecture & LeNet-5**
<!-- #endregion -->

<!-- #region id="tZ5mCvzWpxur" -->
## **CNN Architecture**
<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*7_BCJFzekmPXmJQVRdDgwg.png" width="80%"></center>
<!-- #endregion -->

<!-- #region id="u2rawC7YsNca" -->
Convolutional Neural Networks (CNNs) have been highly successful in various computer vision tasks, such as image classification, object detection, and image segmentation. While CNN architectures can vary based on specific requirements and tasks, a typical CNN architecture consists of several key components. Here is a generic overview of a CNN architecture:

1. **Input Layer:**
   - Accepts the input data, usually in the form of images. The size of the input layer corresponds to the dimensions of the input images.

2. **Convolutional Layers:**
   - Convolutional layers are the core building blocks of CNNs. They use filters (kernels) to convolve over input feature maps, extracting hierarchical features.
   - Activation functions (e.g., ReLU) introduce non-linearity.

3. **Pooling Layers:**
   - Pooling layers follow convolutional layers and downsample the spatial dimensions of feature maps. Common pooling types include Max Pooling and Average Pooling.
   - Pooling contributes to translation invariance and reduces computational complexity.

4. **Flatten Layer:**
   - Flatten layers transition from convolutional layers to fully connected layers. They reshape the 3D output of the convolutional/pooling layers into a 1D vector.
   
5. **Fully Connected Layers:**
   - Fully connected layers connect every neuron to every neuron in the previous and next layers. These layers capture global patterns and relationships in the data.

6. **Output Layer:**
   - The output layer produces the final predictions. The number of neurons in this layer corresponds to the number of classes in a classification task.
   - Activation functions vary based on the task (e.g., softmax for classification).

This architecture is a sequential stack of layers and is commonly implemented using deep learning frameworks such as TensorFlow or PyTorch. Specific CNN architectures like LeNet-5, AlexNet, VGGNet, GoogLeNet (Inception), ResNet, and others have been influential in shaping the field. The choice of architecture depends on factors like the complexity of the task, available data, and computational resources. Researchers often modify and adapt existing architectures or design custom architectures to address specific challenges.
<!-- #endregion -->

<!-- #region id="Iq7keaaFssmM" -->
## **LeNet-5 Architecture**
LeNet-5 is a pioneering convolutional neural network architecture designed for handwritten digit recognition. It was introduced by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner in 1998. LeNet-5 played a crucial role in demonstrating the effectiveness of deep learning in computer vision tasks. <br><br>
<center><img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/Screenshot-from-2021-03-18-12-52-17.png"></center>
<br>

Below is an overview of the LeNet-5 architecture:

1. **Input Layer:**
   - Accepts grayscale images of size 32x32 pixels.

2. **Convolutional Layer (C1):**
   - Convolution with 6 filters of size 5x5.
   - Activation function: Sigmoid.
   - Output feature maps: 28x28x6.

3. **Subsampling Layer (S2):**
   - Average pooling over non-overlapping 2x2 regions.
   - Output feature maps: 14x14x6.

4. **Convolutional Layer (C3):**
   - Convolution with 16 filters of size 5x5.
   - Activation function: Sigmoid.
   - Output feature maps: 10x10x16.

5. **Subsampling Layer (S4):**
   - Average pooling over non-overlapping 2x2 regions.
   - Output feature maps: 5x5x16.

6. **Convolutional Layer (C5):**
   - Convolution with 120 filters of size 5x5.
   - Activation function: Sigmoid.
   - Output feature maps: 1x1x120.

7. **Fully Connected Layer (F6):**
   - Fully connected layer with 84 neurons.
   - Activation function: Sigmoid.

8. **Output Layer:**
   - Fully connected layer with 10 neurons (corresponding to the 10 digits in digit recognition tasks).
   - Activation function: Softmax.

The architecture incorporates a series of convolutional and subsampling layers, followed by fully connected layers. Sigmoid activation functions were used in the original design, and average pooling was employed in the subsampling layers. LeNet-5 demonstrated the effectiveness of deep learning in pattern recognition tasks and laid the foundation for subsequent developments in convolutional neural networks.

It's important to note that while LeNet-5 was groundbreaking at the time, more recent CNN architectures, such as those used in image classification tasks today, often involve deeper networks, different activation functions (e.g., ReLU), and other architectural innovations.
<!-- #endregion -->

<!-- #region id="rroGAqo5vqKx" -->
## **Implementation of LeNet-5 Architecture in Keras**
<!-- #endregion -->

<!-- #region id="QKfOBsgIwdgV" -->
### **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HZj4KksAwAfa" outputId="87794a1d-73b6-4e22-bac4-1b270768aea8"
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.datasets import mnist
print(tf.__version__)
```

<!-- #region id="bTl6KfiRwdI2" -->
### **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zWQgFT_2wjU1" outputId="af1b51da-8bc0-48ad-f3dd-e371e280ae97"
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

<!-- #region id="IMmKh5DAww_Q" -->
### **Build the `LeNet-5` Architecture**
<!-- #endregion -->

```python id="0kOoXdYuw2mw"
# Build the LeNet-5 architecture
model = Sequential()

model.add(Conv2D(6, kernel_size=(5, 5), padding="valid", activation="tanh", input_shape=(32, 32, 1)))
model.add(AveragePooling2D((2, 2), strides=2, padding="valid"))

model.add(Conv2D(16, kernel_size=(5, 5), padding="valid", activation="tanh"))
model.add(AveragePooling2D((2, 2), strides=2, padding="valid"))

model.add(Flatten())

model.add(Dense(120, activation="tanh"))
model.add(Dense(84, activation="tanh"))
model.add(Dense(10, activation="softmax"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="dWT-qUF5yGwT" outputId="e0d8a006-e373-443a-e412-3883e8cc694c"
model.summary()
```

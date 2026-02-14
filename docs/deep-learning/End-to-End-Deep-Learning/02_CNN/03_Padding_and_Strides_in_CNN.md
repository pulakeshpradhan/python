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
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/02_CNN/03_Padding_and_Strides_in_CNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="AEzY8_kXTtiJ" -->
# **Padding and Strides in CNN**
<!-- #endregion -->

<!-- #region id="H2W2Vx7uT29o" -->
## **What is Padding?**

Padding is a technique used to preserve spatial information during the convolutional and pooling operations. It involves adding extra pixels (usually with a value of zero) around the borders of an input feature map or image.

The main purposes of padding are:

1. **Preserving Spatial Information:**
   - Without padding, the spatial dimensions of the feature map decrease with each convolutional layer, potentially leading to a significant reduction in information at the edges.
   - Padding helps maintain the spatial size, ensuring that information near the borders is given proper consideration.

2. **Mitigating the Loss of Information:**
   - In the absence of padding, the pixels at the edges of the feature map are involved in fewer convolution operations, leading to a loss of information.
   - Padding ensures that each pixel in the input has the opportunity to be the center of the receptive field for convolutional filters.

3. **Handling Stride and Filter Size:**
   - Padding becomes especially useful when using larger filter sizes or strides greater than 1. Without padding, the spatial size reduction becomes more pronounced.
<!-- #endregion -->

<!-- #region id="FsR6b6F7T7bq" -->
<center><img src="https://miro.medium.com/v2/resize:fit:1358/1*D6iRfzDkz-sEzyjYoVZ73w.gif" width="70%"></center>
<!-- #endregion -->

<!-- #region id="Cc1fkL9GXE5N" -->
## **Types of Padding in Keras**
In Keras, a popular deep learning library, you can specify different types of padding for convolutional layers. The main types of padding available in Keras are:

1. **Valid Padding (No Padding):**
   - This is the default setting in Keras.
   - No padding is added to the input feature map.
   - The convolution operation is applied only to the valid part of the input.

<center><img src="https://upload.wikimedia.org/wikipedia/commons/7/78/Valid-padding-convolution.gif" width="30%"></center>

2. **Same Padding (Zero Padding):**
   - Padding is added to the input feature map to ensure that the spatial dimensions of the output feature map remain the same as the input.
   - The padding is distributed evenly on all sides.
   - Useful for preserving spatial information and handling larger filter sizes.

<center><img src="https://miro.medium.com/v2/resize:fit:679/1*SsKCClCa9xVxIoaocVY6Ww.gif" width="60%"></center>
<!-- #endregion -->

<!-- #region id="QmzlhxiSY0Ux" -->
## **Calculation of Feature Map Size**
If the stride $(\text{Stride})$ is set to 1 (meaning no skipping of pixels during the convolution), the formula for calculating the feature map size after padding simplifies further. For "same" padding, the formula becomes:

$$\text{Output Size} = {{\text{Input Size} + 2 \times \text{Padding} - \text{Filter Size} + 1}}$$

Here, the terms are the same as in the previous formula:

- $\text{Output Size}$: The size of the feature map after the convolution operation with padding and stride set to 1.
- $\text{Input Size}$: The size of the input (or previous layer's feature map).
- $\text{Filter Size}$: The size of the convolutional filter (kernel).
- $\text{Padding}$: The number of zero-padding pixels added to the input on each side.
<!-- #endregion -->

<!-- #region id="XRvWFJMjZuPX" -->
## **Implementation of Padding in Keras**
<!-- #endregion -->

<!-- #region id="X_VhDSljbGN3" -->
### **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FNfPCR7sZzlY" outputId="7f434a71-6d4e-4769-9e59-0c783c61adb9"
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.datasets import mnist
print(tensorflow.__version__)
```

<!-- #region id="kiB8qW3zbJ4A" -->
### **Read the Data**
<!-- #endregion -->

```python id="oiGjkWo3bOrI"
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

<!-- #region id="GpyQH4m_bj6k" -->
### **Build the Model Architecture with `valid` Padding**
<!-- #endregion -->

```python id="W5XaAUCmbvw8"
# Build the model architecture with 'valid' padding in the convolution layers
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu"))
model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="H9B2_Kr_h4Ls" outputId="b023cf54-25a4-4ab0-8357-72ecf777166b"
# Print the model's summary
model.summary()
```

<!-- #region id="46nf_dlci9sk" -->
### **Build the Model Architecture with `same/zero` Padding**
<!-- #endregion -->

```python id="NhUEadZajEXr"
# Build the model architecture with 'zero' padding in the convolution layers
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="GR1cxM2GkKaL" outputId="d22b576a-47eb-433f-8881-4890795b9cee"
# Print the model's summary
model.summary()
```

<!-- #region id="-GlP0wjpkaiN" -->
## **What is Strides?**
In the context of CNNs, "strides" refer to the step size or the number of pixels the convolutional filter (kernel) moves at each step during the convolution operation. The stride parameter determines the distance between consecutive applications of the filter to the input, influencing the spatial dimensions of the output feature map. `Strided convolution` involves using a stride value greater than 1, meaning that the convolutional filter moves more than one pixel at a time while scanning the input.

Key points about strides:

1. **Stride Value:**
   - Strides are usually set as positive integers.
   - Common values are 1, indicating that the filter moves one pixel at a time, and 2, indicating that the filter moves two pixels at a time.
   - Larger stride values result in a more aggressive reduction of the spatial dimensions.

2. **Effect on Output Size:**
   - Increasing the stride reduces the spatial dimensions of the output feature map.
   - Smaller strides lead to larger feature maps but may increase computational complexity.

3. **Strides and Subsampling:**
   - Strides can be used to achieve subsampling or down-sampling by skipping pixels during the convolution.
   - Subsampling can be beneficial for reducing the computational load and focusing on important features.
<!-- #endregion -->

<!-- #region id="znt1YU9HntY5" -->
**Example of a Convolution Operation when the Stride is set to 2:**
<br>
<center><img src="https://miro.medium.com/v2/resize:fit:679/0*0LMdR2rvJAlRHC3m.gif" width="40%"></center>
<!-- #endregion -->

<!-- #region id="DnNqCyknkgim" -->
## **Calculation of Feature Map Size**
The effect of strides on the output size can be described by the following formula:

$$\text{Output Size} = \frac{\text{Input Size} + 2 \times \text{Padding} - \text{Filter Size}}{{\text{Stride}}} + 1$$

Here are the terms in the formula:

- $\text{Output Size}$: The size of the feature map after the convolution operation.
- $\text{Input Size}$: The size of the input (or previous layer's feature map).
- $\text{Filter Size}$: The size of the convolutional filter (kernel).
- $\text{Padding}$: The number of zero-padding pixels added to the input on each side.
- $\text{Stride}$: The step size or the number of pixels the filter moves at each step during convolution.
<!-- #endregion -->

<!-- #region id="0fKWpnHYpGey" -->
## **Why Strides are required?**
Strides in convolutional neural networks (CNNs) are required for several reasons:

1. **Downsampling and Efficiency:**
   - Strides enable downsampling of the input, reducing the spatial dimensions of the feature maps.
   - Downsampling is crucial for efficiency, reducing computational complexity and memory requirements.

2. **Feature Extraction:**
   - Larger strides skip pixels during convolution, allowing the network to focus on more significant features and patterns.
   - This can be beneficial for capturing high-level features and reducing the spatial size of the feature maps.

3. **Control Over Model Complexity:**
   - Strides provide a way to control the complexity of the model by influencing the spatial dimensions of the feature maps.
   - They allow practitioners to balance between capturing fine-grained details and computational efficiency.

In summary, strides are essential for controlling the trade-off between computational efficiency and feature representation in CNNs. They allow practitioners to tailor the network architecture to the specific requirements of the task at hand, ensuring effective feature extraction and model efficiency.
<!-- #endregion -->

<!-- #region id="Rnwu6HhxpczD" -->
## **Implementation of Strides in Keras**
<!-- #endregion -->

```python id="-S7iEUyupiYd"
# Build the model architecture with a (2, 2) strides in the convolution layers
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding="same", activation="relu"))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding="same", activation="relu"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="DIsVx4g5qWvD" outputId="58a35826-4714-432a-9b20-d3c0dc219be0"
# Print the model's summary
model.summary()
```

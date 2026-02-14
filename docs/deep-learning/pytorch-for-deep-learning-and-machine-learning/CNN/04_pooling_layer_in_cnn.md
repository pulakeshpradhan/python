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

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/02_CNN/04_Pooling_Layer_in_CNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="4EjIIuBbRSwz" -->
# **Pooling Layer in CNN**
<!-- #endregion -->

<!-- #region id="jagaFOm9UGr-" -->
## **The Problem with Convolution**
The convolution operation, while highly effective in capturing spatial hierarchies and features in data, comes with certain challenges. Two notable challenges associated with convolution are **memory issues** and **translation variance**.

The convolution operation, while highly effective in capturing spatial hierarchies and features in data, comes with certain challenges. Two notable challenges associated with convolution are memory issues and translation variance.

1. **Memory Issues:**
   - **High Computational Cost:**
     Convolutional operations, especially with large filter sizes and deep networks, can be computationally expensive. The number of parameters and computations increases, leading to higher memory and processing requirements.

     <center><img src="https://indoml.files.wordpress.com/2018/03/convolution-with-multiple-filters2.png" width="60%"></center>

2. **Translation Variance:**
   - **Positional Sensitivity:**
     Traditional convolution is sensitive to the absolute position of features in the input. This means that the network may not recognize an object if it appears in a different position within the input.
   - **Lack of Invariance:**
     Convolutional networks may lack translation invariance, meaning that slight shifts or translations in the input can significantly affect the network's ability to recognize patterns.
     
     <center><img src="https://i.stack.imgur.com/ylhMz.png" width="60%"></center>
<!-- #endregion -->

<!-- #region id="qieqLcWRULqW" -->
## **What is Pooling?**
Pooling, in CNNs, is a down-sampling operation that reduces the spatial dimensions of the input feature maps while retaining essential information. Pooling is typically applied after convolutional layers to progressively reduce the spatial size, decrease the computational load, and capture the most important features.

Two common types of pooling operations are used in CNNs:

1. **Max Pooling:**
   - Max pooling involves selecting the maximum value from a group of neighboring pixels in the input feature map.

2. **Average Pooling:**
   - Average pooling calculates the average value of a group of neighboring pixels in the input feature map.

Key points about pooling:

- **Spatial Reduction:**
  Pooling reduces the spatial dimensions of the feature map, effectively downsampling the information.

- **Translation Invariance:**
  Pooling contributes to translation invariance by selecting the most relevant features and reducing sensitivity to small shifts or translations in the input.
<br>

**Demo:** [Click on this link.](https://deeplizard.com/resource/pavq7noze3)
<!-- #endregion -->

<!-- #region id="ffYrwhDMXfJz" -->
<center><img src="https://miro.medium.com/v2/resize:fit:679/1*fXxDBsJ96FKEtMOa9vNgjA.gif" width="60%"></center>
<!-- #endregion -->

<!-- #region id="u14sjqqEb4FZ" -->
## **Pooling on Volumes**

<center><img src="https://indoml.files.wordpress.com/2018/03/convolution-with-multiple-filters2.png" width="60%"></center>
<!-- #endregion -->

<!-- #region id="L8rlTiLVcV6k" -->
## **Implementation of Pooling in Keras**
<!-- #endregion -->

<!-- #region id="jNOu0x-Yce_U" -->
### **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="N6P9czPHclyl" outputId="b2afcc81-8b44-40ce-e85b-eb8ce3ae3e6b"
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
print(tf.__version__)
```

<!-- #region id="yZBFyQQAdB_P" -->
### **Read the Data**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yNSrQsoNdGeZ" outputId="64309f2a-c4b1-4a1a-aa29-67656b189757"
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

<!-- #region id="BQoaejfzdXHr" -->
### **Build the Model Architecture with `MaxPooling` Layer**
<!-- #endregion -->

```python id="Nn_7UddRdjHs"
# Build the model architecture with 'MaxPooling' layer
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
```

```python colab={"base_uri": "https://localhost:8080/"} id="9H3oo9X4e0Qo" outputId="712dfcb4-af55-4cd6-d116-92b7e17bed51"
model.summary()
```

<!-- #region id="eQ0NRkVDfu05" -->
### **Advantages of Pooling**
Pooling operations offer several key advantages that contribute to the network's effectiveness in image processing tasks:

1. **Spatial Dimension Reduction:**
   - Pooling reduces the spatial dimensions of input feature maps, effectively downsampling the information. This spatial reduction is crucial for managing computational load and improving efficiency in subsequent layers.
   <center><img src="https://production-media.paperswithcode.com/methods/MaxpoolSample2.png" width="50%"></center>

2. **Translation Invariance:**
   - Pooling introduces a level of translation invariance by selecting the most relevant features. This means the network becomes less sensitive to small shifts or translations in the input, enhancing its ability to generalize across different positions of objects.
   <center><img src="https://i.stack.imgur.com/fTrKfl.png" width="50%"></center>

3. **Enhanced Features:**
   - Pooling operations contribute to enhancing distinctive features within the input data. By selecting the most significant values from local regions, pooling helps the network focus on crucial patterns and characteristics, facilitating better feature extraction.
   <center><img src="https://blog.paperspace.com/content/images/2022/07/compare_pooling_edg-1.png" width="50%"></center>

4. **No Need for Training:**
    - Pooling is a parameter-free operation, requiring no additional training or learnable weights. This simplicity makes it easy to implement and incorporate into CNN architectures without introducing extra parameters that would need to be adjusted during the training process. The lack of trainable parameters in pooling layers adds to the efficiency and simplicity of the overall network design.
<!-- #endregion -->

<!-- #region id="3HtY31R4ix8G" -->
## **Types of Pooling**
There are several types of pooling operations used in Convolutional Neural Networks (CNNs), each serving different purposes in feature extraction and spatial dimension reduction. Three common types of pooling are MaxPooling, AveragePooling, and Global Pooling:

1. **Max Pooling:**
   - **Operation:** Selects the maximum value from a group of neighboring pixels in the input feature map.
   - **Purpose:** Emphasizes the most prominent features, contributing to translation invariance and retaining salient information.
   - **Example:**
     ```
     Input:
     [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]]

     Max Pooling (2x2):
     [[5]]
     ```

2. **Average Pooling:**
   - **Operation:** Calculates the average value from a group of neighboring pixels in the input feature map.
   - **Purpose:** Smooths the representation, reduces sensitivity to outliers, and provides a form of translation invariance.
   - **Example:**
     ```
     Input:
     [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]]

     Average Pooling (2x2):
     [[3.5]]
     ```

3. **Global Pooling (Global Average Pooling or Global Max Pooling):**
   - **Operation:** Computes a single value (global average or global maximum) for each channel across the entire feature map.
   - **Purpose:** Aggregates information globally, reducing the spatial dimensions to a single value per channel. Commonly used as a transition to fully connected layers in classification tasks.
   - **Example:**
     ```
     Input:
     [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]]

     Global Average Pooling:
     [[5]]
     ```
<!-- #endregion -->

<!-- #region id="B-O9RSkijQuI" -->
## **Disadvantages of Pooling**
While pooling operations offer several advantages in Convolutional Neural Networks (CNNs), they also come with certain disadvantages. Here are some drawbacks associated with pooling:

1. **Loss of Spatial Information:**
   - Pooling involves down-sampling, leading to a reduction in spatial dimensions. This reduction can result in the loss of fine-grained spatial information, making it challenging to reconstruct the exact spatial arrangement of features.

2. **Reduced Sensitivity to Small-Scale Patterns:**
   - Max pooling, in particular, focuses on the most significant features within a local region. While this is beneficial for translation invariance, it can lead to reduced sensitivity to small-scale patterns, especially if the maximum values dominate the features.

3. **Not Suitable for Image Segmentation:**
   - In tasks such as image segmentation, where preserving spatial information is crucial, pooling may not be as suitable. Downsampling through pooling can result in a loss of fine-grained details, making it challenging for the network to precisely delineate object boundaries in segmented images.

   <center><img src="https://deeplobe.ai/wp-content/uploads/2023/06/Image-Segmentation_-What-Are-the-Most-Interesting-Applications_.png" width="50%"></center>
<!-- #endregion -->

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
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/02_CNN/02_Convolution_Operation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="1QNzmwG-cWgG" -->
# **Convolution Operation**
<!-- #endregion -->

<!-- #region id="TdRa-xTCcfUf" -->
## **Introduction**
Convolutional Neural Networks (CNNs) consist of multiple layers, each designed to perform specific operations to extract hierarchical features from input data, especially images. The key layers in a CNN include:

1. **Input Layer:**
   - Represents the raw data, often an image with pixel values.
   - The size of the input layer corresponds to the dimensions of the input data, such as the height, width, and number of color channels.

2. **Convolutional Layer:**
   - Applies convolutional operations to the input data using filters or kernels.
   - Each filter captures specific features or patterns within local receptive fields.
   - Activations from this layer form feature maps that represent learned features.

3. **Pooling (Subsampling) Layer:**
   - Performs downsampling to reduce spatial dimensions and computational complexity.
   - Common pooling operations include max pooling or average pooling, which retain the most salient features within local regions.

4. **Fully Connected (Dense) Layer:**
   - Neurons in this layer are connected to all neurons from the previous layer.
   - Transforms high-level features into predictions or class scores.
   - Commonly used in the final layers of the network.

5. **Output Layer:**
   - Produces the final output of the network.
   - The number of neurons in this layer corresponds to the number of classes in a classification task.
   - The activation function is chosen based on the nature of the task (e.g., softmax for classification).
<!-- #endregion -->

<!-- #region id="zpk2ikPsdjv4" -->
<center><img src="https://editor.analyticsvidhya.com/uploads/89175cnn_banner.png" width="75%"></center>
<!-- #endregion -->

<!-- #region id="vXDpyAA_eUfz" -->
## **Basics of Image**
In CNNs, images are fundamental inputs that undergo processing through various layers to extract hierarchical features. The nature of the image, whether grayscale or RGB (Red, Green, Blue), affects the input dimensions and the way the network processes the information.

#### **Grayscale Images:**

1. **Representation:**
   - Grayscale images are represented using a single channel, where each pixel has a single intensity value (typically ranging from 0 to 255).
   - In CNNs, a grayscale image is usually treated as a 2D matrix, where each element represents the intensity of a pixel.

2. **Input Dimensions:**
   - For a grayscale image of size H x W, the input tensor to the CNN would have dimensions (H, W, 1).
   - The third dimension (1) signifies the single channel.

#### **RGB Images:**

1. **Representation:**
   - RGB images are represented using three channels (Red, Green, Blue), where each channel represents the intensity of a specific color.
   - Each pixel has three intensity values, forming a 3D matrix.

2. **Input Dimensions:**
   - For an RGB image of size H x W, the input tensor to the CNN would have dimensions (H, W, 3).
   - The three channels correspond to Red, Green, and Blue.
<!-- #endregion -->

<!-- #region id="fNVRCFZgewvH" -->
<center><img src="https://www.baeldung.com/wp-content/uploads/sites/4/2022/09/NumericalRep.png" width="60%"></center>
<!-- #endregion -->

<!-- #region id="6SoqTCEtfEvx" -->
## **Edge Detection (Convolutional Operation)**
Edge detection is a fundamental operation in image processing and computer vision that aims to identify boundaries within an image. One common approach to edge detection involves using convolutional filters, such as the Sobel filters, to highlight changes in intensity that correspond to edges. The Sobel filters are particularly effective for detecting edges in both the vertical and horizontal directions.
<!-- #endregion -->

<!-- #region id="2gQgUNUyg_N_" -->
#### **Example of a Vertical Edge Detection**

<center><img src="https://media5.datahacker.rs/2018/11/conc_3_1.png" width="70%"></center>
<!-- #endregion -->

<!-- #region id="b78GVQr1hf6J" -->
**Demo:**
[Click on this link.](https://deeplizard.com/resource/pavq7noze2)

<center><img src="https://deeplizard.com/assets/jpg/a74bc5d5.jpg" width="60%"></center>
<!-- #endregion -->

<!-- #region id="OAblCrzbIkRm" -->
## **Calculation of Feature Map Size**
If the stride is set to 1 (meaning no skipping of pixels during the convolution), the formula for calculating the feature map size after a convolution operation simplifies to:

$$ \text{Output Size} = \text{Input Size} - \text{Filter Size} + 1 $$

Here, the terms are:

- $\text{Output Size}$: The size of the feature map after the convolution operation.
- $\text{Input Size}$: The size of the input (or previous layer's feature map).
- $\text{Filter Size}$: The size of the convolutional filter (kernel).

Example:
$\text{Output Size} = \text{Input Size} - \text{Filter Size} + 1 $

Suppose you have a grayscale image with an input size of 28x28 and a filter size of 3x3:

$\text{Output Size} = 28 - 3 + 1 = 26 $

So, in this case, the feature map size after the convolution operation would be 26x26. If you are not using any stride $\text{Stride} = 1 $, this simplified formula is applicable.
<!-- #endregion -->

<!-- #region id="oYqDtjBQJNp2" -->
<center><img src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_3.png" width="70%"></center>
<!-- #endregion -->

<!-- #region id="NH_CBC_VQKuJ" -->
## **Working with RGB Images**
<!-- #endregion -->

<!-- #region id="eUcZBf7NRbyE" -->
**Convolution Operation on RGB Images:**
<center><img src = "https://media5.datahacker.rs/2018/11/06_04.png" width="70%"></center>
<!-- #endregion -->

<!-- #region id="6aV6k9nAR56l" -->
**Applying Multiple Filters on same RGB Image:**
<br>
<center><img src="https://media5.datahacker.rs/2018/11/06_09.png" width="70%"></center>
<!-- #endregion -->

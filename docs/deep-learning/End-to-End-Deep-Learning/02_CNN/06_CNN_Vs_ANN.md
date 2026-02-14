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
<a href="https://colab.research.google.com/github/geonextgis/End-to-End-Deep-Learning/blob/main/02_CNN/06_CNN_Vs_ANN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="GQYvZDsWfnqI" -->
# **CNN Vs ANN**
Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs) are both types of neural networks, but they differ in their architectures, designs, and applications. Here's a brief comparison between CNNs and ANNs:
<!-- #endregion -->

<!-- #region id="Zf6BWNhsrQ2f" -->
**Working Principle of ANN:** <br>
<center><img src="https://adatis.co.uk/wp-content/uploads/historic/HughFreestone_clip_image012.jpg" width="70%"></center>
<!-- #endregion -->

<!-- #region id="7qusjgkKr2Ls" -->
**Working Principle of CNN** <br>
<center><img src="https://miro.medium.com/v2/resize:fit:1400/1*uAeANQIOQPqWZnnuH-VEyw.jpeg" width="70%"></center>
<!-- #endregion -->

<!-- #region id="YhXWKaF4f_UQ" -->
## **Similarities between ANN and CNN**
The working principles of Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) share several fundamental concepts, despite their architectural differences. Here are the key similarities in their working principles:

1. **Neurons and Activation Functions:**
   - Both ANNs and CNNs are composed of interconnected neurons.
   - Neurons in both architectures process weighted inputs, apply activation functions, and produce output signals.
   - Activation functions introduce non-linearity, allowing the network to learn complex relationships in the data.

2. **Forward Propagation:**
   - Both architectures employ forward propagation to process input data and generate predictions.
   - During forward propagation, input signals are passed through the network's layers, and weighted sums are computed at each neuron. The output is then obtained through the activation function.

3. **Loss Function and Training:**
   - Both ANNs and CNNs are trained using a supervised learning approach.
   - They use a loss function to measure the difference between predicted and actual outputs. The goal is to minimize this loss during training.

4. **Backpropagation:**
   - Both architectures use backpropagation as a learning algorithm.
   - Backpropagation involves calculating gradients of the loss with respect to the network's parameters and adjusting these parameters to minimize the loss.

5. **Optimization Algorithms:**
   - Both ANNs and CNNs employ optimization algorithms, such as gradient descent and its variants (e.g., stochastic gradient descent), to update weights and biases during training.
   - These algorithms aim to find the optimal set of parameters that minimize the loss function.

6. **Batch Processing:**
   - Both architectures can process input data in batches during training to improve computational efficiency.
   - **Batch processing involves updating weights and biases based on the average gradient calculated over a batch of input samples.
<!-- #endregion -->

<!-- #region id="QLD-Q6Lut15v" -->
### **Quick Quiz**
1. **How many learnable parameters are there in a convolutional layer with 50 filters applied to an RGB image, where each filter has a shape of (3, 3, 3)?**

    **Ans:**<br>
    Total learnable parameters in a single filter = **`(3 * 3 * 3) + 1 = 28`** <br>
    where, (3, 3, 3) is the shape of a filter and 1 is for bias.<br>
    Total learnable parameters of 50 filters = **`(28 * 50) = 1400`**
<!-- #endregion -->

<!-- #region id="RmuU0aBowLWp" -->
## **Differences between ANN and CNN**
The computational cost of Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs) varies significantly when working with image data. Here are the key differences in terms of computational cost:

1. **Parameter Efficiency:**
    - Fully connected layers in ANNs have a large number of parameters, especially when working with high-dimensional data like images. The sheer number of parameters increases the computational cost during both training and inference.
    - CNNs leverage parameter sharing through convolutional filters, leading to a more parameter-efficient architecture. The use of shared filters significantly reduces the number of parameters compared to ANNs when processing image data.

2. **Localized Operations:**
     - ANN operates on the entire input space, making it computationally intensive, especially for large images. Lacks the ability to efficiently capture localized patterns.
     - CNN Employs localized operations, such as convolution and pooling, which significantly reduce the computational cost. Focuses on specific regions of the input, enhancing efficiency in capturing local patterns.

3. **Hierarchical Feature Extraction:**
     - Hierarchical feature extraction in ANNs involves fully connected layers, which may struggle to capture spatial hierarchies effectively. Computationally expensive due to the large number of parameters and global connectivity.
     - CNNs are designed for hierarchical feature extraction through convolutional layers. Filters capture spatial hierarchies locally, making them computationally efficient in processing image data.

4. **Spatial Invariance:**
     - ANN lacks inherent spatial invariance, requiring extensive processing to handle variations in object position within an image. Higher computational cost in dealing with spatial transformations.
     - CNN incorporates pooling layers to achieve spatial invariance, reducing the computational burden related to position variations. Robustness to translations results in computational efficiency in handling diverse spatial configurations.
<!-- #endregion -->

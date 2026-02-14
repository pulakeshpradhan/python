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
<a href="https://colab.research.google.com/github/geonextgis/PyImgProc-Image-Processing-using-Python/blob/main/21_waterbodies_extraction_using_entropy_and_otsu's_threshold.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="OXlIe2bvjK5V" -->
# **Water Bodies Extraction using Entropy and Otsu's Threshold**
<!-- #endregion -->

<!-- #region id="lRTwbUkpHNZ6" -->
## **Import Required Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hJlqQPvjGkT2" outputId="90152821-2da3-493a-94a4-d7dffd2eb138"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="rgrFS-LQGxtY"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu

plt.rcParams["font.family"] = "DejaVu Serif"
```

<!-- #region id="LlNd6RSrG6lC" -->
## **Load the Image**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HfyXF7h7HRm4" outputId="5e493ec5-d240-4592-db90-d2dd59b48667"
# Read the image with skimage
img_path = "/content/drive/MyDrive/Colab Notebooks/GitHub Repo/PyImgProc-Image-Processing-using-Python/Datasets/High_Res_RGB_Google_Image.tif"
img = io.imread(img_path)
print(img.shape)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="nu1lBHenHM7u" outputId="20a5bcd7-759f-4753-9dfc-4eb38b93e6f6"
# Plot the image
plt.imshow(img);
```

<!-- #region id="AkfYrYO1K9w5" -->
## **Applying Entropy Filter**
In image processing, an entropy filter is a type of spatial filter used to enhance or detect features in an image based on the concept of entropy. Entropy, in the context of image processing, measures the amount of randomness or uncertainty in the distribution of pixel values within a neighborhood of the image.

#### **How Entropy Filter Works**

1. **Neighborhood Definition**: Similar to other spatial filters, an entropy filter operates by sliding a window or kernel across the image. This window defines a local neighborhood around each pixel.

2. **Entropy Calculation**: Within each neighborhood, the entropy filter computes the entropy of the pixel values. Entropy is calculated using the histogram of pixel intensities within the neighborhood. The formula for entropy calculation is often based on Shannon's entropy formula:

   $$H = -\sum_{i} P(i) \log_2 P(i)$$

   where $P(i)$ represents the probability of occurrence of pixel value $i$ within the neighborhood. The sum is taken over all possible pixel values.

3. **Filter Response**: The calculated entropy value for each neighborhood is used to determine the response of the filter. Typically, high entropy values indicate regions with high variability or complexity, while low entropy values indicate regions with more uniform intensity distributions.

4. **Enhancement or Detection**: Depending on the application, the entropy filter may enhance regions with high entropy (e.g., edges, textures) or suppress regions with low entropy (e.g., uniform regions, noise).

#### **Applications of Entropy Filter**

1. **Texture Analysis**: Entropy filters are often used for texture analysis in images. Regions with high entropy correspond to complex textures, while regions with low entropy correspond to smoother textures.

2. **Edge Detection**: Entropy filters can be used for edge detection since edges often correspond to regions with high variability in pixel values.

3. **Image Segmentation**: Entropy-based segmentation methods use entropy filters to identify regions with distinct texture or intensity characteristics.

4. **Noise Reduction**: By suppressing regions with low entropy, entropy filters can help reduce noise in an image.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="p1FnftbzOJxr" outputId="b10b44f2-6ee0-4ad5-a298-1b1f4e65cd50"
# Apply the entropy filter on red channel of the RGB image
entropy_img = entropy(img[:, :, 0], footprint=disk(3))

# Plot the entropy image
plt.imshow(entropy_img)
plt.colorbar();
```

<!-- #region id="odzIdXACVYfm" -->
## **Applying Otsu's Thresholding**
Otsu's thresholding is a global thresholding technique used in image processing to automatically perform clustering-based image thresholding. Named after Nobuyuki Otsu, who introduced it in 1979, the method is used to convert a grayscale image into a binary image by finding an optimal threshold that minimizes the intra-class variance or equivalently maximizes the inter-class variance.

#### **How Otsu's Thresholding Works**

1. **Histogram Calculation**: Compute the histogram of the grayscale image, which represents the frequency of each gray level (intensity).

2. **Probability Distribution**: Normalize the histogram to obtain the probability distribution of each gray level.

3. **Class Probabilities and Means**:
   - For each possible threshold $( t )$:
     - Compute the class probabilities:
       $$
       \omega_0(t) = \sum_{i=0}^{t-1} P(i) \quad \text{(probability of class 1)}
       $$
       $$
       \omega_1(t) = \sum_{i=t}^{L-1} P(i) \quad \text{(probability of class 2)}
       $$
     - Compute the class means:
       $$
       \mu_0(t) = \frac{\sum_{i=0}^{t-1} i \cdot P(i)}{\omega_0(t)}
       $$
       $$
       \mu_1(t) = \frac{\sum_{i=t}^{L-1} i \cdot P(i)}{\omega_1(t)}
       $$
    

4. **Intra-Class Variance Calculation**: Calculate the intra-class variance for each threshold $( t )$:
   $$
   \sigma^2_w(t) = \omega_0(t) \sigma^2_0(t) + \omega_1(t) \sigma^2_1(t)
   $$

   where $ \sigma^2_0(t) $ and $ \sigma^2_1(t) $ are the variances of the two classes, computed as:
   $$
   \sigma^2_0(t) = \sum_{i=0}^{t-1} (\mu_0(t) - i)^2 P(i)
   $$

   $$
   \sigma^2_1(t) = \sum_{i=t}^{L-1} (\mu_1(t) - i)^2 P(i)
   $$

5. **Optimal Threshold Selection**: The optimal threshold $( t^* )$ is the one that minimizes the intra-class variance (or equivalently, maximizes the inter-class variance):
   $$
   t^* = \arg \min_{t} \sigma^2_w(t)
   $$

#### **Applications**

- **Image Binarization**: Converting grayscale images to binary images for applications such as document scanning, medical imaging, and industrial inspection.
- **Object Segmentation**: Separating objects from the background in an image.
- **Preprocessing**: Preparing images for further analysis by reducing complexity.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="mW63KNUlVlH3" outputId="13a081db-227a-4f56-8163-5914adf6e247"
# Plot the histogtam of the entropy image
sns.histplot(entropy_img.flatten());
```

```python colab={"base_uri": "https://localhost:8080/"} id="DbYuhXhUcS0G" outputId="7dc4903d-9134-42fa-aa9c-549b2a68cf88"
# Apply Otsu's thresholding on entropy image
# Get the Otsu's threshold value
thresh = threshold_otsu(entropy_img)
print("Otsu's threshold value:", thresh)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="W2b4L0WGczyC" outputId="2c9b7412-be83-4e75-a764-3b4fb7acd74e"
# Create a binary image
binary_img = entropy_img <= thresh

# Plot the binary image
plt.imshow(binary_img);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="TPixfl4jeYAr" outputId="255e3b47-d3d3-48cf-e266-edf2c43508f5"
# Plot the Otsu's threshold on the histogram
plt.figure()

ax = sns.histplot(entropy_img.flatten())
plt.axvline(x=thresh, c="red", linestyle="--", label=f"Otsu's Threshold\n({round(thresh, 4)})")
plt.legend();
```

<!-- #region id="s5z-weNNf4sS" -->
## **Plot all the Images in a Single Layout**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 397} id="m42vgLsJgEwT" outputId="6dc9d568-a976-4a8f-afe9-945bc0fffd0c"
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axes = axes.flatten()

axes[0].imshow(img)
axes[0].set_title("RGB Image")

axes[1].imshow(entropy_img)
axes[1].set_title("Entropy Image")

axes[2].imshow(binary_img)
axes[2].set_title("Binary Image (Otsu's thresholded)");
```

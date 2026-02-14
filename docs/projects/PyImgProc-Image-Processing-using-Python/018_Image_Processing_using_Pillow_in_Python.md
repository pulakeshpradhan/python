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
<a href="https://colab.research.google.com/github/geonextgis/PyImgProc-Image-Processing-using-Python/blob/main/018_Image_Processing_using_Pillow_in_Python.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="GSzgY4AehkD4" -->
# **Image Processing using Pillow in Python**
PIL stands for Python Imaging Library. It's a library in Python that adds support for opening, manipulating, and saving many different image file formats. PIL provides functionalities such as image resizing, cropping, filtering, and basic image processing operations.

PIL has been a popular choice for working with images in Python for many years. However, its development was discontinued after version 1.1.7. Fortunately, the Pillow library was created as a fork of PIL to continue its development and maintenance. Pillow is essentially a drop-in replacement for PIL, providing all the functionalities of PIL and more, while also being actively maintained and updated. Therefore, when people refer to PIL today, they often mean Pillow.

Pillow (or PIL) is widely used in various Python projects for tasks such as image manipulation, computer vision, web development, and more.
<!-- #endregion -->

<!-- #region id="yruPRK4FBTeo" -->
## **Import Necessary Libraries**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IwZFMtReh0h5" outputId="894e71d8-f7ab-4565-a548-99d7c7c6e3a2"
from google.colab import drive
drive.mount("/content/drive")
```

```python id="-lcnxX7sAY-B"
from PIL import Image
import glob
```

<!-- #region id="7SGGN5JpBdKf" -->
## **Load an Image**
<!-- #endregion -->

<!-- #region id="sqwsrgdSjsbP" -->
🤔 **Note:**<br> The mode of an image defines the type and depth of a pixel in the image. The current release supports the following standard modes:

- 1 (1-bit pixels, black and white, stored with one pixel per byte)

- L (8-bit pixels, black and white)

- P (8-bit pixels, mapped to any other mode using a colour palette)

- RGB (3x8-bit pixels, true colour)

- RGBA (4x8-bit pixels, true colour with transparency mask)

- CMYK (4x8-bit pixels, colour separation)

- YCbCr (3x8-bit pixels, colour video format)

- I (32-bit signed integer pixels)

- F (32-bit floating point pixels)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="q0Wf66r3icS-" outputId="8841ce98-efac-4ad4-c8ad-7a2e3b70b669"
img_path = "/content/drive/MyDrive/Colab Notebooks/GitHub Repo/PyImgProc-Image-Processing-using-Python/Datasets/High_Res_RGB_Google_Image.tif"
img = Image.open(img_path)

# Print the type of the image
print("Image Type:", type(img))

# Print the format of the image
print("Image Format:", img.format)

# Print the mode of the image
print("Image Mode:", img.mode)

# Print the size of the image
print("Image Size:", img.size)
```

<!-- #region id="gDJwQs4qrbUk" -->
## **Resizing an Image**
In the Python Imaging Library (PIL), which is now maintained as the Pillow library, you can resize an image using the resize() method. Additionally, the thumbnail() method provides a way to resize an image while preserving its aspect ratio.

- In the `resize(` method, you specify the new dimensions for the image. If the aspect ratio of the new dimensions doesn't match the aspect ratio of the original image, the resized image will be distorted.

- In contrast, the `thumbnail()` method resizes the image to fit within the specified dimensions while preserving the aspect ratio. The image is resized so that the longer side fits within the given dimensions, and the other side is adjusted accordingly. This means the image is resized proportionally and not distorted.

<!-- #endregion -->

<!-- #region id="5N4m44pyIRBZ" -->
### **Resize**
<!-- #endregion -->

```python id="jA5JY6Vtrd78"
img_resized = img.resize((200, 300))
img_resized.save("test_image_resize.jpg")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 334} id="6TqFJU2bGK5k" outputId="8a979af2-744d-469a-e443-1efe2450f314"
print(img_resized.size)
display(img_resized)
```

```python id="Ml4nziNTH72H"
img_resized = img.resize((4048, 4600))
img_resized.save("test_image_resize.jpg")
```

```python colab={"base_uri": "https://localhost:8080/"} id="X2VvfNHgIP-g" outputId="a242c636-d67a-49b7-c22f-4ce024a23396"
print(img_resized.size)
# display(img_resized)
```

<!-- #region id="R0Llxkl8IUFZ" -->
### **Thumbnail**
<!-- #endregion -->

```python id="6pIFKBDBGfVW"
img.thumbnail((200, 300))
img.save("test_image_thumbnail.jpg")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 234} id="ewZsX7YlHDDi" outputId="27f42c77-15b6-49a5-a6dd-292184a5966b"
print(img.size)
display(img)
```

```python id="GBJqCmhwHt-F"
# img.thumbnail((1200, 1200))
# img.save("test_image_thumbnail.jpg")
```

```python id="JCYCyXzRH0zm"
# print(img.size)
# display(img)
```

<!-- #region id="yT2SRvQuIkyr" -->
## **Cropping an Image**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 184} id="xGUujgavIo2K" outputId="1b413426-da2c-4ac7-b2e0-4b7f4a2d83e7"
cropped_image = img.crop((0, 0, 150, 150))
cropped_image.save(f"cropped_image.jpg")

print(cropped_image.size)
display(cropped_image)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 234} id="PwQJhElnJ5Wy" outputId="62c2ba58-6945-4ed7-fa15-fec9ae7a71b9"
# Copy and Paste images
img_google = Image.open("/content/google.png")

img_copy = img.copy()

# Paste the google image on top of the copied image
img_copy.paste(img_google, (10, 10))
img_copy.save("Pasted_Image.jpg")

print(img_copy.size)
display(img_copy)
```

<!-- #region id="MTTJpkPVLdH9" -->
## **Image Rotation**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 234} id="OINqbiRWLf94" outputId="4e8b8bfb-f727-4e3e-8c5e-494a7333e487"
# Rotate the image about 90 degrees
img_90 = img.rotate(90)
img_90.save("rotated_image_90.jpg")

print(img_90.size)
display(img_90)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 234} id="jT3PPLpDLzt5" outputId="491edf17-34dd-45a8-e35c-20379d058609"
# Rotate the image about 45 degrees
img_45 = img.rotate(45)
img_45.save("rotated_image_45.jpg")

print(img_45.size)
display(img_45)
```

<!-- #region id="ge0LhiSiMChT" -->
## **Image Flip**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 234} id="2p1YTqVmMFVL" outputId="bfbace2f-e79d-47f6-f276-d62a1f79abd3"
# Flip the image Left to Right
image_flipLR = img.transpose(Image.FLIP_LEFT_RIGHT)
image_flipLR.save("flipped_image_LR.jpg")

print(image_flipLR.size)
display(image_flipLR)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 234} id="PoBhIHohMQ85" outputId="eb8f6b59-007e-427f-9bab-bff11fe0c0e3"
# Flip the image Top to Bottom
image_flipTB = img.transpose(Image.FLIP_TOP_BOTTOM)
image_flipTB.save("flipped_image_TB.jpg")

print(image_flipTB.size)
display(image_flipTB)
```

<!-- #region id="weLRYD1BMtgn" -->
## **Grayscale**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 234} id="6WyATkOKMyFw" outputId="ee270efc-2d9e-47cc-abe8-49eb4f8d3e25"
# Convert the image into grayscale
gray_img = img.convert("L")
gray_img.save("grayscale_image.jpg")

print(gray_img.size)
display(gray_img)
```

<!-- #region id="vPGqEG2nOvwD" -->
## **Automate Image Processing Task using glob**
<!-- #endregion -->

```python id="2Sv3-OYLO5o1"
folder_path = "/content/drive/MyDrive/Colab Notebooks/GitHub Repo/PyImgProc-Image-Processing-using-Python/Datasets/*.tif"

# Read all the tif files and save a 45 degree rotated version in jpg format
for f in glob.glob(folder_path):
    temp_img = Image.open(f)
    temp_img_rotated = temp_img.rotate(45)
    # Convert into Grayscale
    temp_img_rotated = temp_img_rotated.convert("L")
    file_name = f.split("/")[-1].split(".")[0] + "_Rotated.jpg"
    temp_img_rotated.save(file_name, "PNG")
```

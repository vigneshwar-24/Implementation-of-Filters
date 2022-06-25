# Exp: 06
# Implementation-of-Filters
## AIM:
To implement filters for smoothing and sharpening the images in the spatial domain.

## SOFTWARE REQUIRED:
Anaconda - Python 3.7

## ALGORITHM:
### Step 1:
Import the necessary modules.

### Step 2:
For performing smoothing operation on a image.

Average filter
```
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
```
Weighted average filter
```
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
```
Gaussian Blur
```
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
```
Median filter
```
median=cv2.medianBlur(image2,13)
```
### Step 3:
For performing sharpening on a image.

Laplacian Kernel
```
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
```
Laplacian Operator
```
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
```
### Step 4:
Display all the images with their respective filters.

## PROGRAM:
```
# Developed By   : Vigneshwar s
# Register Number: 212220230058

import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("tae.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
```
### 1. Smoothing Filters
i) Using Averaging Filter
```
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```
ii) Using Weighted Averaging Filter
```
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```
iii) Using Gaussian Filter
```
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
iv) Using Median Filter
```
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```
### 2. Sharpening Filters
i) Using Laplacian Kernal
```
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()
```
ii) Using Laplacian Operator
```
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```
## OUTPUT:
### 1. Smoothing Filters
i) Using Averaging Filter

<img width="355" alt="d1" src="https://user-images.githubusercontent.com/77089276/167682826-0c14d8f6-ced7-48df-9631-d58d5c7234b5.PNG">

ii) Using Weighted Averaging Filter

<img width="363" alt="d2" src="https://user-images.githubusercontent.com/77089276/167682904-d404a9a7-2cb3-454b-b934-f689d0ceab41.PNG">

iii) Using Gaussian Filter

<img width="352" alt="d3" src="https://user-images.githubusercontent.com/77089276/167682922-453f592e-d3ec-4f55-a814-fc9139bfd34c.PNG">

iv) Using Median Filter

<img width="356" alt="d4" src="https://user-images.githubusercontent.com/77089276/167682944-9708a1ba-ba65-4fc9-88c3-0d8ff4139ee3.PNG">

### 2. Sharpening Filters
i) Using Laplacian Kernal

<img width="354" alt="d5" src="https://user-images.githubusercontent.com/77089276/167683031-be4fdaea-40a2-451e-9f8a-d58278ca97d4.PNG">

ii) Using Laplacian Operator

<img width="351" alt="d6" src="https://user-images.githubusercontent.com/77089276/167683093-5c620c4d-2c22-4906-bbd1-aa982c2bb45c.PNG">

## RESULT:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.

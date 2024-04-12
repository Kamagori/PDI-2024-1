import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from skimage.io import imread
from skimage.filters import roberts, sobel, scharr, prewitt

img = cv2.imread("aviao.jpg", 0)

# Filtro Sobel
sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
filtered_image_x = cv2.convertScaleAbs(sobelx)

sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
filtered_image_y = cv2.convertScaleAbs(sobely)

sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
filtered_image_xy = cv2.convertScaleAbs(sobelxy)
plt.figure(figsize=(18, 19))
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis("off")

plt.subplot(222)
plt.imshow(filtered_image_x, cmap='gray')
plt.title('Sobel X')
plt.axis("off")

plt.subplot(223)
plt.imshow(filtered_image_y, cmap='gray')
plt.title('Sobel Y')
plt.axis("off")

plt.subplot(224)
plt.imshow(filtered_image_xy, cmap='gray')
plt.title('Sobel X Y')
plt.axis("off")
plt.show()

# prewitt
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img, -1, kernelx)
img_prewitty = cv2.filter2D(img, -1, kernely)
img_prewitt = img_prewittx + img_prewitty
cv2.imshow("Filtro Preewit", img_prewitt)
cv2.imwrite("prewitt.png", img_prewittx)
cv2.waitKey()

# roberts


img = cv2.imread("aviao.jpg", 0).astype('float64')
img /= 255.0

op_roberts = roberts(img)

cv2.imshow("Roberts", op_roberts)
cv2.imwrite("roberts.jpg", op_roberts)
cv2.waitKey(0)

# implementando 5 metodos de binarização do opencv

def otsu_threshold(image):
    # Inicializa as variáveis
    pixel_counts = Counter(image.flatten())
    pixel_intensities = np.array(list(pixel_counts.keys()))
    total_pixels = sum(pixel_counts.values())
    sum_intensities = np.sum(pixel_intensities * np.array([pixel_counts[i] for i in pixel_intensities]))
    max_variance = 0
    threshold = 0

    # Loop para calcular o limiar de Otsu
    for t in range(len(pixel_intensities)):
        # Calcula a probabilidade de cada classe (background e foreground)
        w1 = sum([pixel_counts[pixel_intensities[i]] for i in range(t)]) / total_pixels
        w2 = 1 - w1

        # Calcula as médias das intensidades de cada classe
        sum1 = sum([pixel_intensities[i] * pixel_counts[pixel_intensities[i]] for i in range(t)])
        mean1 = sum1 / (total_pixels * w1) if w1 != 0 else 0
        sum2 = sum_intensities - sum1
        mean2 = sum2 / (total_pixels * w2) if w2 != 0 else 0

        # Calcula a variância interclasse
        variance = w1 * w2 * ((mean1 - mean2) ** 2)

        # Atualiza o valor do limiar se a variância for maior que a anterior
        if variance > max_variance:
            max_variance = variance
            threshold = pixel_intensities[t]

    return threshold



threshold = 0
max_value = 255

image = cv2.imread("aviao.jpg", 0)
threshold = otsu_threshold(image)
print(threshold)
# when applying OTSU threshold, set threshold to 0.

_, output1 = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, output2 = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, output3 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
_, output4 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
_, output5 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

images = [image, output1, output2, output3, output4, output5]
titles = ["Orignals", "Binary", "Binary Inverse", "TOZERO", "TOZERO INV", "TRUNC"]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.show()

image = cv2.imread("prewitt.png", 0)
threshold = otsu_threshold(image)
print(threshold)
# when applying OTSU threshold, set threshold to 0.

_, output1 = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, output2 = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, output3 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
_, output4 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
_, output5 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

images = [image, output1, output2, output3, output4, output5]
titles = ["Orignals", "Binary", "Binary Inverse", "TOZERO", "TOZERO INV", "TRUNC"]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.show()

image = cv2.imread("roberts.jpg", 0)
threshold = otsu_threshold(image)
print(threshold)
# when applying OTSU threshold, set threshold to 0.

_, output1 = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, output2 = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, output3 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
_, output4 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
_, output5 = cv2.threshold(image, threshold, max_value, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

images = [image, output1, output2, output3, output4, output5]
titles = ["Orignals", "Binary", "Binary Inverse", "TOZERO", "TOZERO INV", "TRUNC"]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.show()






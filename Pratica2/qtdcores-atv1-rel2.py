import numpy as np
import cv2
img = cv2.imread('halteres.jpg')
Z = img.reshape((-1,3))
# Transformar a imagem em uma lista de tuplas (r, g, b)
tuplas_unicas = set(map(tuple, Z))
# Contar quantas tuplas únicas foram encontradas
num_cor_pixel_utilizada = len(tuplas_unicas)
print("Esta imagem utilizou %d combinações de cores RGB" % num_cor_pixel_utilizada)
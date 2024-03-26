import easyocr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re

def detectar_placa(image, ocr):
    # Realizar OCR na imagem
    result = ocr.readtext(image)
    placa = None
    # Exibir os resultados
    placas = []
    print(f"Texto                              Probabilidade")
    for detection in result:
        text = detection[1]
        probabilidade = detection[2]
        print(f"{text:<35}{probabilidade:.4f}")
        
        if (len(text) == 7):
            placas.append(text)
    if len(placas) == 1:
        placa = placas[0]
    else:
        padrao = r'[A-Za-z]{3}\d[A-Za-z]\d{2}'
        for texto in placas:
            if re.findall(padrao, texto):
                placa = texto
    return placa

def limiarizacao_otsu(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_thresh, otsu_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu_img

def clusterizacao_kmeans(image, k):
    # Converte a imagem em um array NumPy de tipo float32
    data = np.float32(image).reshape((-1, 3))

    # Aqui, é definido o critério de parada para o algoritmo K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    # Execução do kmeans
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Os centroides são convertidos para o tipo de dados uint8.
    center = np.uint8(center)
    # Replace pixel values with their center value:
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result

# Carregar o modelo de OCR para placa
ocr = easyocr.Reader(['pt'])
placa = None

# Carregar a imagem da placa
image = cv2.imread('images/placa1.png')
print("Deteccao com OCR da imagem original\n")
placa = detectar_placa(image=image, ocr=ocr)
print(f"A placa detectada foi: {placa}")
print("\n------------------------------------------------------------------")

image_otsu = limiarizacao_otsu(image)
print("Deteccao com OCR da imagem apos aplicar limiarizacao\n")
placa = detectar_placa(image=image_otsu, ocr=ocr)
print(f"A placa detectada foi: {placa}")
print("\n------------------------------------------------------------------")

image_kmens = clusterizacao_kmeans(image, 3)
print("Deteccao com OCR da imagem apos aplicar clusterizacao\n")
placa = detectar_placa(image=image_kmens, ocr=ocr)
print(f"A placa detectada foi: {placa}")


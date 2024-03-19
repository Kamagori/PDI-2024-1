
import cv2

# carrega a imagem em tons de cinza
gray_img = cv2.imread('estatua.png',0)
# define o tamanho da janela de equalização local
win_size = (32, 32)
# aplica a equalização local do histograma
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=win_size)
equalized_img = clahe.apply(gray_img)
# mostra a imagem original e a imagem equalizada
cv2.imshow('Imagem original', gray_img)
cv2.imshow('Imagem Equalizda com clahe', equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
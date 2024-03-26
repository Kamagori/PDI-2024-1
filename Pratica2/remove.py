import cv2
import numpy as np
from skimage.filters import threshold_local

video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame_prev = cap.read()
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
contador = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])  # aplica equalização no canal V

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular a diferença entre os frames consecutivos
    diff = cv2.absdiff(gray, gray_prev)

    niblack = cv2.ximgproc.niBlackThreshold(diff, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=15, k=-0.2)

    adaptive = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, gaussian = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    radius = 15
    local_thresh = threshold_local(diff, block_size=radius, offset=10)
    wellner = diff > local_thresh

    cv2.imshow("Niblack", niblack)
    cv2.imshow("Adaptive", adaptive)
    cv2.imshow("Gaussian", gaussian)
    cv2.imshow("WELLNER", wellner.astype(np.uint8) * 255)


    cv2.imshow("Original Frame", frame)
    cv2.waitKey(100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Atualizar o frame anterior
    gray_prev = gray

cap.release()
cv2.destroyAllWindows()
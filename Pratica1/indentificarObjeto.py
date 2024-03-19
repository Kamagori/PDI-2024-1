import cv2
import numpy as np

def detect_object(image, lower_color, upper_color):
    # Threshold para detectar a cor azul
    mask = cv2.inRange(image, lower_color, upper_color)
    # Aplicar a máscara na imagem original
    res = cv2.bitwise_and(image, image, mask=mask)

    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Se houver contornos encontrados
    if contours:
        # Encontrar o contorno com a maior área
        max_contour = max(contours, key=cv2.contourArea)
        
        # Calcular o centro do contorno
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return res, (cx, cy)
    
    return res, None

# Ler o vídeo
cap = cv2.VideoCapture(0)

# Definir intervalo de cores azuis (em BGR)
lower_blue = np.array([120, 0, 0])
upper_blue = np.array([255, 125, 70])

lower_color = lower_blue
upper_color = upper_blue

# Variável para armazenar o caminho percorrido pelo objeto
caminho = []
altura = largura = 0
# Defina o caminho em vermelho
cor_vermelha = (0, 0, 255)  # BGR
image = ""
# Loop para processar cada frame do vídeo
while(cap.isOpened()):
    ret, frame = cap.read()
    # Inverter horizontalmente
    frame = cv2.flip(frame, 1)

    altura, largura , canais= frame.shape
    if ret == True:

        image, object_position = detect_object(frame, lower_color, upper_color)

        # Se a posição do objeto for encontrada, adicione ao caminho
        if object_position:
            caminho.append(object_position)

        # Desenhar a linha vermelha conectando os pontos do caminho
        if len(caminho) > 1:
            for i in range(1, len(caminho)):
                cv2.line(image, caminho[i-1], caminho[i], cor_vermelha, 2)

        # Exibir o frame com a linha vermelha
        cv2.imshow('Red Path', image)
        
        # Pressione 'q' para sair
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Liberar o objeto VideoCapture e fechar as janelas
cap.release()
cv2.destroyAllWindows()

# Salve a imagem como um arquivo PNG
cv2.imwrite("images/imagem_com_caminho.png", image)

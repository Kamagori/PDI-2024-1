import cv2
import numpy as np
import matplotlib.pyplot as plt

y_linhas = []
linhas_selecionadas = []  # Lista para armazenar as linhas selecionadas
fechar_imagem = False

def plotar_grafico_variacao_nivel_cinza(dados, y_img):
    for i in range(len(dados)):
        plt.figure(y_img[i])
        plt.plot(dados[i])
        plt.xlabel('Coluna do pixel')
        plt.ylabel('Valor do pixel')
        plt.suptitle(f'Variação do nivel de cinza da linha y = {y_img[i]}')
    plt.show()


def filtro_media_blur(img_original, y_linhas, kernel):

    median=cv2.blur(img_original,ksize=(kernel,kernel))
    cv2.imshow(f"Filtro Median Blur ({kernel},{kernel})" ,median)
    dados = []
    linha = median[y_linhas[0], :]
    dados.append(linha)
    linha = median[y_linhas[1], :]
    dados.append(linha)
    linha = median[y_linhas[2], :]
    dados.append(linha)

    plotar_grafico_variacao_nivel_cinza(dados, y_linhas)
    cv2.waitKey()

def mouse_callback(event, x, y, flags, param):
    global linhas_selecionadas
    global y_linhas
    global fechar_imagem
    if event == cv2.EVENT_RBUTTONDOWN:
        # Obtém a linha correspondente à posição do mouse
        linha = gray[y, :]
        # Adiciona a linha à lista de linhas selecionadas
        linhas_selecionadas.append(linha)
        y_linhas.append(y)
        # Se já selecionamos três linhas, podemos parar de registrar cliques
        if len(linhas_selecionadas) == 3:
            cv2.setMouseCallback('Imagem', lambda *args: None)  # Remove o callback do mouse
            fechar_imagem = True

# Carrega a imagem do disco
img = cv2.imread('Pratica4/images/lena.png')
# Converte a imagem para tons de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Cria uma janela para exibir a imagem
cv2.namedWindow('Imagem')
# Registra a função callback para o evento de clique do mouse
cv2.setMouseCallback('Imagem', mouse_callback)
# Exibe a imagem na janela
cv2.imshow('Imagem', gray)

# Aguarda que o usuário pressione a tecla 'q' para sair ou selecionar 3 linhas
while True:
    if (cv2.waitKey(1) & 0xFF == ord('q')) | (fechar_imagem == True):
        break
# Fecha a janela
cv2.destroyAllWindows()


# Plotar o gráfico 
plotar_grafico_variacao_nivel_cinza(dados=linhas_selecionadas, y_img=y_linhas)

filtro_media_blur(img_original=gray, y_linhas=y_linhas, kernel=3)
filtro_media_blur(img_original=gray, y_linhas=y_linhas, kernel=5)
filtro_media_blur(img_original=gray, y_linhas=y_linhas, kernel=7)


cv2.destroyAllWindows()

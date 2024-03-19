import numpy as np
import cv2

def extract_5_pixels(image, color):
    pixels = []
    cv2.imshow(f"Selecione pixels da cor {color}", image)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixels.append(image[y, x])
            print(f"Pixel {len(pixels)}: {image[y, x]}")

    cv2.setMouseCallback(f"Selecione pixels da cor {color}", on_mouse)

    while len(pixels) < 5:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Calcular a média dos pixels
    avg_bgr = np.mean(pixels, axis=0)

    return avg_bgr


def process_image(image, media, tol, tamanho_janela, mask):
    rows, cols, _ = image.shape
    output_image = np.copy(image)  # Iniciar com uma cópia da imagem original

    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            total_b, total_g, total_r = 0, 0, 0

            # Verificar se o pixel atual não foi colorido e se pertence ao objeto
            if np.any(mask[i, j]):
                for m in range(i - 2, i + 3):
                    for n in range(j - 2, j + 3):
                        total_b += image[m, n][0]
                        total_g += image[m, n][1]
                        total_r += image[m, n][2]

                avg_b = total_b / 25
                avg_g = total_g / 25
                avg_r = total_r / 25

                # Verificar se a média da cor do quadrado 5x5 está na tolerância
                if ((media[0] - tol) < avg_b < (media[0] + tol)) and \
                   ((media[1] - tol) < avg_g < (media[1] + tol)) and \
                   ((media[2] - tol) < avg_r < (media[2] + tol)):
                    output_image[i-2:i+3, j-2:j+3] = media

    return output_image


def fill_remaining_pixels(image, colors, tol):
    rows, cols, _ = image.shape

    for i in range(2, rows - 2, 5):
        for j in range(2, cols - 2, 5):
            total_b, total_g, total_r = 0, 0, 0
            total_pixels = 0

            # Calcular a média dos pixels no quadrado 5x5
            for m in range(i - 2, i + 3):
                for n in range(j - 2, j + 3):
                    total_b += image[m, n][0]
                    total_g += image[m, n][1]
                    total_r += image[m, n][2]
                    total_pixels += 1

            avg_b = total_b / total_pixels
            avg_g = total_g / total_pixels
            avg_r = total_r / total_pixels

            # Verificar se a média da cor do quadrado 5x5 está dentro da tolerância
            for color, media in colors.items():
                if ((media[0] - tol) < avg_b < (media[0] + tol)) and \
                   ((media[1] - tol) < avg_g < (media[1] + tol)) and \
                   ((media[2] - tol) < avg_r < (media[2] + tol)):
                    # Verificar se o pixel central é mais claro que a média
                    if image[i, j][0] > avg_b and image[i, j][1] > avg_g and image[i, j][2] > avg_r:
                        image[i - 2:i + 3, j - 2:j + 3] = media

    return image


def save_image(image, filename):
    cv2.imwrite(filename, image)
    print(f"Imagem salva como '{filename}'")

img = cv2.imread("halteres.jpg")

# Inicializar o dicionário de cores com valores vazios que serão preenchidos com as médias
colors = {'Azul': [], 'Preto': [], 'Amarelo': [],
          'Vermelho': [], 'Verde': [], 'Prata': []}

output_image = np.copy(img)  # Iniciar com uma cópia da imagem original

# Inicializar a máscara
mask = np.ones_like(img[:, :, 0], dtype=bool)

# Processar cada cor e colorir na imagem de saída
for color in ['Azul', 'Preto', 'Amarelo', 'Vermelho', 'Verde', 'Prata']:
    # Extrair os pixels e calcular as médias para cada cor
    avg_bgr = extract_5_pixels(output_image, color)
    colors[color] = avg_bgr

    # Processar a cor atual na imagem de saída
    output_image = process_image(output_image, colors[color], 50, 5, mask)

    # Atualizar a máscara para evitar repintura
    mask &= np.all(output_image != colors[color], axis=-1)


output_image = fill_remaining_pixels(output_image, {color: colors[color]}, 50)

save_image(output_image, "halteres_final.jpg")

cv2.imshow("Final", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


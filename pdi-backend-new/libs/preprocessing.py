import cv2


def converter_para_cinza(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def aplicar_filtro_gaussiano(imagem, tamanho_kernel=(5, 5)):
    return cv2.GaussianBlur(imagem, tamanho_kernel, 0)


def detectar_bordas_canny(imagem, limiar_baixo=100, limiar_alto=200):
    return cv2.Canny(imagem, limiar_baixo, limiar_alto)

import cv2
import numpy as np

def segmentar_objeto_com_flood_fill(imagem):
    _, mascara = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = mascara.shape
    ff = mascara.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(ff, ff_mask, seedPoint=seed, newVal=0)
    
    kernel = np.ones((3, 3), np.uint8)
    ff = cv2.morphologyEx(ff, cv2.MORPH_OPEN, kernel, iterations=1)
    ff = cv2.morphologyEx(ff, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return ff

def filtrar_contornos_borda(contornos, largura_imagem, altura_imagem, margem=1):
    def toca_borda(x, y, w, h, W, H, m=1):
        return x <= m or y <= m or (x + w) >= (W - m) or (y + h) >= (H - m)
    
    contornos_filtrados = []
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if not toca_borda(x, y, w, h, largura_imagem, altura_imagem, margem):
            contornos_filtrados.append(contorno)
    
    return contornos_filtrados

def encontrar_contornos(mascara_binaria):
    contornos, _ = cv2.findContours(mascara_binaria.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def desenhar_contornos(imagem, contornos, cor=(0, 255, 0), espessura=2):
    imagem_contornos = imagem.copy()
    cv2.drawContours(imagem_contornos, contornos, -1, cor, espessura)
    return imagem_contornos

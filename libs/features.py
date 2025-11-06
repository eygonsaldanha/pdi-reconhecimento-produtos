import cv2
import numpy as np
from libs.segmentation import segmentar_simples


def extrair_histograma_rgb(imagem, mascara, bins=32):
    if imagem is None or mascara is None:
        return np.zeros(bins * 3)
    
    try:
        imagem_mascarada = cv2.bitwise_and(imagem, imagem, mask=mascara)
        
        hist_b = cv2.calcHist([imagem_mascarada], [0], mascara, [bins], [0, 256])
        hist_g = cv2.calcHist([imagem_mascarada], [1], mascara, [bins], [0, 256])
        hist_r = cv2.calcHist([imagem_mascarada], [2], mascara, [bins], [0, 256])
        
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        
        hist_rgb = np.concatenate([hist_r, hist_g, hist_b])
        
        return hist_rgb
    
    except:
        return np.zeros(bins * 3)


def extrair_histograma_hsv(imagem, mascara, bins=32):
    if imagem is None or mascara is None:
        return np.zeros(bins * 3)
    
    try:
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        hsv_mascarado = cv2.bitwise_and(hsv, hsv, mask=mascara)
        
        hist_h = cv2.calcHist([hsv_mascarado], [0], mascara, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv_mascarado], [1], mascara, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv_mascarado], [2], mascara, [bins], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        hist_hsv = np.concatenate([hist_h, hist_s, hist_v])
        
        return hist_hsv
    
    except:
        return np.zeros(bins * 3)


def processar_imagem_completa(imagem_colorida):
    if imagem_colorida is None:
        return None
    
    if not isinstance(imagem_colorida, np.ndarray):
        return None
    
    if len(imagem_colorida.shape) != 3:
        return None
    
    h, w = imagem_colorida.shape[:2]
    if h < 10 or w < 10:
        return None
    
    try:
        gray = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
        
        mascara, contorno = segmentar_simples(gray)
        
        if mascara is None or contorno is None:
            return None
        
        area = cv2.contourArea(contorno)
        if area < (h * w * 0.01):
            return None
        
        hist_rgb = extrair_histograma_rgb(imagem_colorida, mascara, bins=32)
        hist_hsv = extrair_histograma_hsv(imagem_colorida, mascara, bins=32)
        
        vetor = np.concatenate([
            hist_rgb,
            hist_hsv
        ])
        
        vetor = np.nan_to_num(vetor, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(vetor) != 192:
            return None
        
        return vetor
    
    except Exception as e:
        return None

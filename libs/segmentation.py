import cv2
import numpy as np


def segmentar_simples(imagem_cinza):
    if imagem_cinza is None or len(imagem_cinza.shape) != 2:
        return None, None
    
    try:
        blur = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        maior_contorno = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(maior_contorno)
        
        h, w = imagem_cinza.shape
        if area < (h * w * 0.01):
            return None, None
        
        mascara = np.zeros_like(imagem_cinza)
        cv2.drawContours(mascara, [maior_contorno], -1, 255, -1)
        
        return mascara, maior_contorno
    
    except:
        return None, None

"""
Módulo simplificado de extração de características.
Usa métodos padrão e comprovados para reconhecimento de imagens.
"""
import cv2
import numpy as np


def segmentar_simples(imagem_cinza):
    """
    Segmentação simples usando Otsu thresholding.
    Retorna máscara e contorno principal.
    """
    if imagem_cinza is None or len(imagem_cinza.shape) != 2:
        return None, None
    
    try:
        # Aplicar blur para suavizar
        blur = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
        
        # Threshold Otsu
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operações morfológicas para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Pegar maior contorno
        maior_contorno = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(maior_contorno)
        
        # Validar área mínima
        h, w = imagem_cinza.shape
        if area < (h * w * 0.01):  # Pelo menos 1% da imagem
            return None, None
        
        # Criar máscara
        mascara = np.zeros_like(imagem_cinza)
        cv2.drawContours(mascara, [maior_contorno], -1, 255, -1)
        
        return mascara, maior_contorno
    
    except:
        return None, None


def extrair_histograma_rgb(imagem, mascara, bins=32):
    """
    Extrai histograma RGB normalizado.
    """
    if imagem is None or mascara is None:
        return np.zeros(bins * 3)
    
    try:
        # Aplicar máscara
        imagem_mascarada = cv2.bitwise_and(imagem, imagem, mask=mascara)
        
        # Calcular histogramas para cada canal
        hist_b = cv2.calcHist([imagem_mascarada], [0], mascara, [bins], [0, 256])
        hist_g = cv2.calcHist([imagem_mascarada], [1], mascara, [bins], [0, 256])
        hist_r = cv2.calcHist([imagem_mascarada], [2], mascara, [bins], [0, 256])
        
        # Normalizar
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        
        # Concatenar
        hist_rgb = np.concatenate([hist_r, hist_g, hist_b])
        
        return hist_rgb
    
    except:
        return np.zeros(bins * 3)


def extrair_histograma_hsv(imagem, mascara, bins=32):
    """
    Extrai histograma HSV normalizado.
    """
    if imagem is None or mascara is None:
        return np.zeros(bins * 3)
    
    try:
        # Converter para HSV
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        
        # Aplicar máscara
        hsv_mascarado = cv2.bitwise_and(hsv, hsv, mask=mascara)
        
        # Calcular histogramas
        hist_h = cv2.calcHist([hsv_mascarado], [0], mascara, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv_mascarado], [1], mascara, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv_mascarado], [2], mascara, [bins], [0, 256])
        
        # Normalizar
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenar
        hist_hsv = np.concatenate([hist_h, hist_s, hist_v])
        
        return hist_hsv
    
    except:
        return np.zeros(bins * 3)


def extrair_caracteristicas_forma_simples(contorno):
    """
    Extrai características básicas de forma.
    """
    if contorno is None:
        return np.zeros(4)
    
    try:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        
        # Circularidade: 4*pi*area / perimetro^2
        if perimetro > 0:
            circularidade = (4 * np.pi * area) / (perimetro * perimetro)
        else:
            circularidade = 0
        
        # Aspect ratio (bounding box)
        x, y, w, h = cv2.boundingRect(contorno)
        if h > 0:
            aspect_ratio = float(w) / h
        else:
            aspect_ratio = 0
        
        # Extent: area / bounding_box_area
        if w * h > 0:
            extent = area / (w * h)
        else:
            extent = 0
        
        return np.array([area, perimetro, circularidade, aspect_ratio, extent])
    
    except:
        return np.zeros(5)


def extrair_textura_simples(imagem_cinza, mascara):
    """
    Extrai características de textura simples usando gradientes.
    """
    if imagem_cinza is None or mascara is None:
        return np.zeros(3)
    
    try:
        # Aplicar máscara
        imagem_mascarada = cv2.bitwise_and(imagem_cinza, imagem_cinza, mask=mascara)
        
        # Calcular gradientes Sobel
        sobel_x = cv2.Sobel(imagem_mascarada, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(imagem_mascarada, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude do gradiente
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Estatísticas do gradiente
        pixels_validos = mascara > 0
        if np.any(pixels_validos):
            media = np.mean(magnitude[pixels_validos])
            desvio = np.std(magnitude[pixels_validos])
            maximo = np.max(magnitude[pixels_validos])
        else:
            media = 0
            desvio = 0
            maximo = 0
        
        return np.array([media, desvio, maximo])
    
    except:
        return np.zeros(3)


def processar_imagem_simples(imagem_colorida):
    """
    Processa imagem completa e retorna vetor de características simplificado.
    Usa APENAS histogramas de cor (muito eficazes para reconhecimento de frutas).
    
    Retorna vetor de 192 dimensões:
    - Histograma RGB: 32*3 = 96
    - Histograma HSV: 32*3 = 96
    Total: 192 dimensões
    """
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
        # Converter para cinza
        gray = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
        
        # Segmentação simples
        mascara, contorno = segmentar_simples(gray)
        
        if mascara is None or contorno is None:
            return None
        
        # Validar área mínima
        area = cv2.contourArea(contorno)
        if area < (h * w * 0.01):
            return None
        
        # Extrair APENAS histogramas de cor (muito eficazes para frutas)
        hist_rgb = extrair_histograma_rgb(imagem_colorida, mascara, bins=32)
        hist_hsv = extrair_histograma_hsv(imagem_colorida, mascara, bins=32)
        
        # Concatenar histogramas
        vetor = np.concatenate([
            hist_rgb,      # 96 dimensões
            hist_hsv       # 96 dimensões
        ])
        
        # Remover valores inválidos
        vetor = np.nan_to_num(vetor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validar tamanho
        if len(vetor) != 192:
            return None
        
        return vetor
    
    except Exception as e:
        return None


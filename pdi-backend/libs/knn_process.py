import cv2
import numpy as np
import pandas as pd
from libs.preprocessing import converter_para_cinza, aplicar_filtro_gaussiano, detectar_bordas_canny
from libs.segmentation import segmentar_objeto_com_flood_fill, filtrar_contornos_borda, encontrar_contornos, desenhar_contornos
from libs.geometric import calcular_area, calcular_perimetro, calcular_circularidade, calcular_aspect_ratio
from libs.features import extrair_lbp, extrair_glcm, extrair_hog

def ensure_flatten(x) -> np.ndarray:
    if isinstance(x, dict):
        values = []
        for v in x.values():
            if isinstance(v, (list, np.ndarray, int, float)):
                values.extend(np.ravel(v))
        return np.array(values, dtype=float)
    
    if isinstance(x, (list, tuple)):
        flat = []
        for item in x:
            flat.extend(np.ravel(item) if isinstance(item, (list, np.ndarray)) else [item])
        return np.array(flat, dtype=float)
    
    if isinstance(x, (int, float, np.number)):
        return np.array([x], dtype=float)
    return np.array([0.0], dtype=float)

def knn_process_df_image(image_path=None, image_process=None):
    if image_path != None:
        image_process = cv2.imread(image_path)
    
    img_rgb = cv2.cvtColor(image_process, cv2.COLOR_BGR2RGB)
    img_cinza = converter_para_cinza(image_process)
    img_suavizada = aplicar_filtro_gaussiano(img_cinza)
    img_bordas_canny = detectar_bordas_canny(img_suavizada)
    
    mascara_segmentada = segmentar_objeto_com_flood_fill(img_suavizada)
    contornos = encontrar_contornos(mascara_segmentada)
    altura_img, largura_img = img_cinza.shape
    contornos_filtrados = filtrar_contornos_borda(contornos, largura_img, altura_img)
    mascara_final = np.zeros_like(img_cinza)
    img_com_contornos = desenhar_contornos(img_rgb, contornos_filtrados, cor=(0, 255, 0), espessura=2)
    
    if contornos_filtrados:
        contorno_principal = max(contornos_filtrados, key=cv2.contourArea)
    else:
        contorno_principal = None
    
    metricas_geo = []
    if contorno_principal is not None:
        area = calcular_area(contorno_principal)
        perimetro = calcular_perimetro(contorno_principal)
        circularidade = calcular_circularidade(contorno_principal)
        aspect_ratio = calcular_aspect_ratio(contorno_principal)
        metricas_geo = [area, perimetro, circularidade, aspect_ratio]
        
    vetor_hog, img_visual_hog = extrair_hog(img_cinza) 
    img_visual_lbp, hist_lbp = extrair_lbp(img_cinza)
    metricas_glcm = extrair_glcm(img_cinza)
        
    return {
        'image_process': np.ravel(image_process),
        'img_rgb': np.ravel(img_rgb),
        'img_cinza': np.ravel(img_cinza),
        'img_suavizada': np.ravel(img_suavizada),
        'img_bordas_canny': np.ravel(img_bordas_canny),
        'mascara_segmentada': np.ravel(mascara_segmentada),
        'contornos': ensure_flatten(contornos),
        'altura_img': np.ravel(altura_img),
        'largura_img': np.ravel(largura_img),
        'contornos_filtrados': ensure_flatten(contornos_filtrados),
        'mascara_final': np.ravel(mascara_final),
        'img_com_contornos': np.ravel(img_com_contornos),
        'metricas_geo': np.ravel(metricas_geo),
        'vetor_hog': np.ravel(vetor_hog),
        'img_visual_hog': np.ravel(img_visual_hog),
        'img_visual_lbp': np.ravel(img_visual_lbp),
        'hist_lbp': np.ravel(hist_lbp),
        'metricas_glcm': [metricas_glcm[i] for i in metricas_glcm],
    }
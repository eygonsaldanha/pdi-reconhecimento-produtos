import cv2
import numpy as np
import pandas as pd
from preprocessing import converter_para_cinza, aplicar_filtro_gaussiano, detectar_bordas_canny
from segmentation import segmentar_objeto_com_flood_fill, filtrar_contornos_borda, encontrar_contornos, desenhar_contornos
from geometric import calcular_area, calcular_perimetro, calcular_circularidade, calcular_aspect_ratio
from features import extrair_lbp, extrair_glcm, extrair_hog

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
    contours = cv2.drawContours(mascara_final, contornos_filtrados, -1, (255), -1)
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

    def ensure_flatten(x) -> np.ndarray:
        arr = np.array(x)
        if arr.ndim == 0:
            return np.array([arr])
        elif arr.ndim > 1:
            return arr.flatten()
        return arr

    processed_pdi = [image_process, img_rgb, img_cinza, img_suavizada, img_bordas_canny, mascara_segmentada, contornos, altura_img, largura_img, contornos_filtrados, mascara_final, contours, img_com_contornos, metricas_geo, vetor_hog, img_visual_hog, img_visual_lbp, hist_lbp, metricas_glcm]
    processed_pdi = [ensure_flatten(i) for i in processed_pdi]

    data = np.concatenate(processed_pdi)
    return pd.DataFrame([data], columns=[f'feat_{i}' for i in range(len(data))])
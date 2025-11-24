import cv2
import numpy as np
from typing import Tuple
from libs.features import *
from libs.geometric import *
from libs.hog_features import *
from libs.knn_process import *
from libs.preprocessing import *
from libs.segmentation import *
from libs.visualization import *


def extract_features(
    image: np.ndarray,
    hsv_bins: Tuple[int, int, int] = (8, 8, 8),
    lbp_points: int = 24,
    lbp_radius: int = 3,
    lbp_bins: int = 256,
) -> np.ndarray:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_cinza = converter_para_cinza(image)
    img_suavizada = aplicar_filtro_gaussiano(img_cinza)
    img_bordas_canny = detectar_bordas_canny(img_suavizada)

    mascara_segmentada = segmentar_objeto_com_flood_fill(img_suavizada)
    contornos = encontrar_contornos(mascara_segmentada)
    altura_img, largura_img = img_cinza.shape
    contornos_filtrados = filtrar_contornos_borda(contornos, largura_img, altura_img)
    mascara_final = np.zeros_like(img_cinza)
    img_com_contornos = desenhar_contornos(
        img_rgb, contornos_filtrados, cor=(0, 255, 0), espessura=2
    )

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

    normal_lists = []
    for element in [
        list(image.flatten()),
        list(img_rgb.flatten()),
        list(img_cinza.flatten()),
        list(img_suavizada.flatten()),
        list(img_bordas_canny.flatten()),
        list(mascara_final.flatten()),
        list(img_com_contornos.flatten()),
        list(vetor_hog),
        list(img_visual_hog.flatten()),
        list(img_visual_lbp.flatten()),
        list(hist_lbp),
        # np.concatenate([c1.flatten() for c1 in contornos]).tolist(),
        # np.concatenate([c.reshape(-1) for c in contornos_filtrados]).tolist(),
        [largura_img, altura_img],
        metricas_geo,
        [metricas_glcm[i] for i in metricas_glcm],
    ]:
        aux_list = [float(x) for x in element]
        normal_lists.append(aux_list)

    features = np.concatenate(normal_lists)
    return features


def extract_features_from_dataset(
    images: list,
    hsv_bins: Tuple[int, int, int] = (8, 8, 8),
    lbp_points: int = 24,
    lbp_radius: int = 3,
    lbp_bins: int = 256,
    verbose: bool = True,
) -> np.ndarray:
    features_list = []

    for i, img in enumerate(images):
        if verbose and (i + 1) % 50 == 0:
            print(f"Extraindo características: {i + 1}/{len(images)}")

        features = extract_features(img, hsv_bins, lbp_points, lbp_radius, lbp_bins)
        features_list.append(features)

    if verbose:
        print(f"Extração completa: {len(images)} imagens")

    return np.array(features_list)

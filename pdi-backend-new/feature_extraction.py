"""
Módulo de extração de características usando técnicas de PDI.
Implementa extração de características HSV (cor) e LBP (textura).
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from typing import Tuple


def extract_hsv_histogram(image: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """
    Extrai histograma HSV de uma imagem.
    
    O espaço de cor HSV é mais robusto a variações de iluminação que RGB.
    
    Args:
        image: Imagem BGR (formato OpenCV)
        bins: Número de bins para cada canal (H, S, V)
        
    Returns:
        Vetor de características do histograma HSV normalizado
    """
    # Converte BGR para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calcula histograma 3D
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],  # Canais H, S, V
        None,
        bins,  # Número de bins por canal
        [0, 180, 0, 256, 0, 256]  # Ranges: H[0,180], S[0,256], V[0,256]
    )
    
    # Normaliza o histograma
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist


def extract_lbp_histogram(image: np.ndarray, num_points: int = 8, radius: int = 1, bins: int = 256) -> np.ndarray:
    """
    Extrai características LBP (Local Binary Patterns) de uma imagem.
    
    LBP captura informações de textura locais comparando pixels vizinhos.
    
    Args:
        image: Imagem BGR (formato OpenCV)
        num_points: Número de pontos vizinhos ao redor de cada pixel
        radius: Raio do círculo ao redor do pixel central
        bins: Número de bins para o histograma LBP
        
    Returns:
        Vetor de características do histograma LBP normalizado
    """
    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcula LBP
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    
    # Calcula histograma do LBP
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=bins,
        range=(0, bins),
        density=True
    )
    
    return hist


def extract_features(image: np.ndarray, 
                     hsv_bins: Tuple[int, int, int] = (8, 8, 8),
                     lbp_points: int = 24,
                     lbp_radius: int = 3,
                     lbp_bins: int = 256) -> np.ndarray:
    """
    Extrai características completas combinando HSV e LBP.
    
    Pipeline:
    1. Extrai histograma HSV (características de cor)
    2. Extrai histograma LBP (características de textura)
    3. Concatena os vetores em um único vetor de características
    
    Args:
        image: Imagem BGR pré-processada
        hsv_bins: Bins para histograma HSV
        lbp_points: Número de pontos para LBP
        lbp_radius: Raio para LBP
        lbp_bins: Bins para histograma LBP
        
    Returns:
        Vetor de características concatenado (HSV + LBP)
    """
    # Extrai características HSV
    hsv_features = extract_hsv_histogram(image, bins=hsv_bins)
    
    # Extrai características LBP
    lbp_features = extract_lbp_histogram(image, num_points=lbp_points, radius=lbp_radius, bins=lbp_bins)
    
    # Concatena os vetores de características
    features = np.concatenate([hsv_features, lbp_features])
    
    return features


def extract_features_from_dataset(images: list, 
                                  hsv_bins: Tuple[int, int, int] = (8, 8, 8),
                                  lbp_points: int = 24,
                                  lbp_radius: int = 3,
                                  lbp_bins: int = 256,
                                  verbose: bool = True) -> np.ndarray:
    """
    Extrai características de um conjunto de imagens.
    
    Args:
        images: Lista de imagens pré-processadas
        hsv_bins: Bins para histograma HSV
        lbp_points: Número de pontos para LBP
        lbp_radius: Raio para LBP
        lbp_bins: Bins para histograma LBP
        verbose: Se True, exibe progresso
        
    Returns:
        Array 2D onde cada linha é o vetor de características de uma imagem
    """
    features_list = []
    
    for i, img in enumerate(images):
        if verbose and (i + 1) % 50 == 0:
            print(f"Extraindo características: {i + 1}/{len(images)}")
        
        features = extract_features(img, hsv_bins, lbp_points, lbp_radius, lbp_bins)
        features_list.append(features)
    
    if verbose:
        print(f"Extração completa: {len(images)} imagens")
    
    return np.array(features_list)
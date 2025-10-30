#!/usr/bin/env python3
"""
Módulo de características de textura - LBP e GLCM.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def extrair_lbp(imagem, p=8, r=1):
    """
    Extrai características LBP (Local Binary Pattern) de uma imagem.
    
    Args:
        imagem: Imagem de entrada (numpy.ndarray)
        p: Número de pontos vizinhos (int, default=8)
        r: Raio do padrão (int, default=1)
    
    Returns:
        tuple: (lbp_imagem, lbp_histograma)
    """
    if len(imagem.shape) == 3:
        # Converter para escala de cinza se necessário
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        imagem_cinza = imagem.copy()
    
    # Extrair LBP
    lbp = local_binary_pattern(imagem_cinza, P=p, R=r, method="uniform")
    
    # Normalizar para visualização
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-9)
    lbp_img = (lbp_norm * 255).astype(np.uint8)
    
    # Calcular histograma
    lbp_bins = np.arange(0, lbp.max() + 2)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, density=True)
    
    return lbp_img, lbp_hist


def extrair_glcm(imagem, distancias=[1, 2, 3], angulos=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extrai características GLCM (Gray Level Co-occurrence Matrix) de uma imagem.
    
    Args:
        imagem: Imagem de entrada (numpy.ndarray)
        distancias: Lista de distâncias (list, default=[1, 2, 3])
        angulos: Lista de ângulos em radianos (list, default=[0, π/4, π/2, 3π/4])
    
    Returns:
        dict: Dicionário com características GLCM
    """
    if len(imagem.shape) == 3:
        # Converter para escala de cinza se necessário
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        imagem_cinza = imagem.copy()
    
    # Calcular GLCM
    glcm = graycomatrix(
        imagem_cinza, 
        distances=distancias, 
        angles=angulos, 
        levels=256, 
        symmetric=True, 
        normed=True
    )
    
    # Extrair características
    caracteristicas = {}
    propriedades = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    
    for prop in propriedades:
        caracteristicas[prop] = float(graycoprops(glcm, prop).mean())
    
    return caracteristicas


def extrair_caracteristicas_textura_completas(imagem, lbp_params=None, glcm_params=None):
    """
    Extrai todas as características de textura de uma imagem.
    
    Args:
        imagem: Imagem de entrada (numpy.ndarray)
        lbp_params: Parâmetros LBP (dict, default=None)
        glcm_params: Parâmetros GLCM (dict, default=None)
    
    Returns:
        dict: Dicionário com todas as características de textura
    """
    if lbp_params is None:
        lbp_params = {'P': 8, 'R': 1}
    
    if glcm_params is None:
        glcm_params = {
            'distancias': [1, 2, 3],
            'angulos': [0, np.pi/4, np.pi/2, 3*np.pi/4]
        }
    
    try:
        # Extrair LBP
        lbp_img, lbp_hist = extrair_lbp(imagem, **lbp_params)
        
        # Extrair GLCM
        glcm_caracteristicas = extrair_glcm(imagem, **glcm_params)
        
        # Calcular estatísticas LBP
        lbp_mean = float(np.mean(lbp_hist))
        lbp_std = float(np.std(lbp_hist))
        
        return {
            'lbp': {
                'imagem': lbp_img,
                'histograma': lbp_hist.tolist(),
                'media': lbp_mean,
                'desvio': lbp_std,
                'bins': len(lbp_hist)
            },
            'glcm': glcm_caracteristicas,
            'extraction_success': True
        }
        
    except Exception as e:
        return {
            'lbp': {
                'imagem': None,
                'histograma': [],
                'media': 0.0,
                'desvio': 0.0,
                'bins': 0
            },
            'glcm': {},
            'extraction_success': False,
            'error': str(e)
        }

import cv2
import numpy as np
import math


def calcular_area_objeto(contorno):
    """
    Calcula a área do objeto através do contorno.
    
    Args:
        contorno: Contorno do objeto (numpy.ndarray)
    
    Returns:
        float: Área do objeto em pixels²
    """
    if contorno is None or len(contorno) < 3:
        return 0.0
    
    area = cv2.contourArea(contorno)
    return float(area)


def calcular_perimetro(contorno):
    """
    Calcula o perímetro do objeto através do contorno.
    
    Args:
        contorno: Contorno do objeto (numpy.ndarray)
    
    Returns:
        float: Perímetro do objeto em pixels
    """
    if contorno is None or len(contorno) < 2:
        return 0.0
    
    perimetro = cv2.arcLength(contorno, closed=True)
    return float(perimetro)


def calcular_circularidade(contorno):
    """
    Calcula a circularidade do objeto (0-1).
    
    Args:
        contorno: Contorno do objeto (numpy.ndarray)
    
    Returns:
        float: Circularidade (1 = círculo perfeito, 0 = linha)
    """
    if contorno is None or len(contorno) < 3:
        return 0.0
    
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, closed=True)
    
    if perimetro == 0:
        return 0.0
    
    # Fórmula: 4 * π * área / perímetro²
    circularidade = (4 * math.pi * area) / (perimetro ** 2)
    return float(circularidade)


def calcular_aspect_ratio(contorno):
    """
    Calcula a proporção largura/altura do bounding box.
    
    Args:
        contorno: Contorno do objeto (numpy.ndarray)
    
    Returns:
        float: Aspect ratio (1 = quadrado, >1 = horizontal, <1 = vertical)
    """
    if contorno is None or len(contorno) < 3:
        return 0.0
    
    # Obter bounding box
    x, y, w, h = cv2.boundingRect(contorno)
    
    if h == 0:
        return 0.0
    
    aspect_ratio = w / h
    return float(aspect_ratio)


def extrair_caracteristicas_geometricas_completas(contorno):
    """
    Extrai todas as características geométricas de um contorno.
    
    Args:
        contorno: Contorno do objeto (numpy.ndarray)
    
    Returns:
        dict: Dicionário com todas as características geométricas
    """
    if contorno is None or len(contorno) < 3:
        return {
            'area': 0.0,
            'perimetro': 0.0,
            'circularidade': 0.0,
            'aspect_ratio': 0.0,
            'bounding_box': (0, 0, 0, 0),
            'centroide': (0.0, 0.0)
        }
    
    # Calcular características básicas
    area = calcular_area_objeto(contorno)
    perimetro = calcular_perimetro(contorno)
    circularidade = calcular_circularidade(contorno)
    aspect_ratio = calcular_aspect_ratio(contorno)
    
    # Calcular bounding box
    x, y, w, h = cv2.boundingRect(contorno)
    bounding_box = (int(x), int(y), int(w), int(h))
    
    # Calcular centroide
    M = cv2.moments(contorno)
    if M['m00'] != 0:
        cx = float(M['m10'] / M['m00'])
        cy = float(M['m01'] / M['m00'])
    else:
        cx, cy = 0.0, 0.0
    
    centroide = (cx, cy)
    
    return {
        'area': area,
        'perimetro': perimetro,
        'circularidade': circularidade,
        'aspect_ratio': aspect_ratio,
        'bounding_box': bounding_box,
        'centroide': centroide
    }


def extrair_caracteristicas_geometricas_multiplos_contornos(contornos):
    """
    Extrai características geométricas de múltiplos contornos.
    
    Args:
        contornos: Lista de contornos (list)
    
    Returns:
        list: Lista de dicionários com características de cada contorno
    """
    if not contornos:
        return []
    
    caracteristicas = []
    for contorno in contornos:
        feat = extrair_caracteristicas_geometricas_completas(contorno)
        caracteristicas.append(feat)
    
    return caracteristicas


def filtrar_contornos_por_area(contornos, area_minima=100):
    """
    Filtra contornos por área mínima.
    
    Args:
        contornos: Lista de contornos (list)
        area_minima: Área mínima em pixels² (int)
    
    Returns:
        list: Contornos filtrados
    """
    if not contornos:
        return []
    
    contornos_filtrados = []
    for contorno in contornos:
        area = calcular_area_objeto(contorno)
        if area >= area_minima:
            contornos_filtrados.append(contorno)
    
    return contornos_filtrados


def obter_maior_contorno(contornos):
    """
    Retorna o maior contorno da lista.
    
    Args:
        contornos: Lista de contornos (list)
    
    Returns:
        numpy.ndarray: Maior contorno ou None
    """
    if not contornos:
        return None
    
    maior_contorno = max(contornos, key=lambda c: calcular_area_objeto(c))
    return maior_contorno

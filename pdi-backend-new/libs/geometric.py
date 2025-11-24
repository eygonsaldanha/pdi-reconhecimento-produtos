import cv2
import numpy as np
from typing import Union

def calcular_area(contorno) -> float:
    """
    Calcula a área de um contorno.
    
    Args:
        contorno: Contorno obtido de cv2.findContours
        
    Returns:
        float: Área do contorno em pixels (sempre positiva)
    """
    if contorno is None or len(contorno) == 0:
        return 0.0
    return abs(cv2.contourArea(contorno))  # abs() garante valor positivo

def calcular_perimetro(contorno) -> float:
    """
    Calcula o perímetro de um contorno.
    
    Args:
        contorno: Contorno obtido de cv2.findContours
        
    Returns:
        float: Perímetro do contorno
    """
    if contorno is None or len(contorno) == 0:
        return 0.0
    return cv2.arcLength(contorno, closed=True)

def calcular_circularidade(contorno) -> float:
    """
    Calcula a circularidade de um contorno.
    Circularidade = 4π * área / perímetro²
    Valores próximos a 1 indicam formas mais circulares.
    
    Args:
        contorno: Contorno obtido de cv2.findContours
        
    Returns:
        float: Circularidade (0 a 1, onde 1 é um círculo perfeito)
    """
    if contorno is None or len(contorno) == 0:
        return 0.0
    
    area = abs(cv2.contourArea(contorno))
    perimetro = cv2.arcLength(contorno, closed=True)
    
    if perimetro == 0:
        return 0.0
    
    return (4 * np.pi * area) / (perimetro ** 2)

def calcular_aspect_ratio(contorno) -> float:
    """
    Calcula a proporção de aspecto (largura/altura) do bounding box.
    
    Args:
        contorno: Contorno obtido de cv2.findContours
        
    Returns:
        float: Aspect ratio (largura/altura do retângulo delimitador)
    """
    if contorno is None or len(contorno) == 0:
        return 0.0
    
    x, y, w, h = cv2.boundingRect(contorno)
    
    if h == 0:
        return 0.0
    
    return float(w) / float(h)
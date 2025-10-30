import cv2
import numpy as np
from skimage.feature import hog
from typing import Dict, Tuple, List, Optional


def extrair_hog_basico(imagem, orientacoes=9, pixels_por_celula=(8, 8), celulas_por_bloco=(2, 2)):
    """
    Extrai características HOG básicas de uma imagem.
    
    Args:
        imagem: Imagem de entrada (numpy.ndarray)
        orientacoes: Número de orientações (int, default=9)
        pixels_por_celula: Tamanho da célula em pixels (tuple, default=(8,8))
        celulas_por_bloco: Número de células por bloco (tuple, default=(2,2))
    
    Returns:
        tuple: (hog_vector, hog_imagem_visualizacao)
    """
    if len(imagem.shape) == 3:
        # Converter para escala de cinza se necessário
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        imagem_cinza = imagem.copy()
    
    # Extrair HOG
    hog_vector, hog_imagem = hog(
        imagem_cinza,
        orientations=orientacoes,
        pixels_per_cell=pixels_por_celula,
        cells_per_block=celulas_por_bloco,
        visualize=True,
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    # Normalizar imagem de visualização
    hog_imagem_norm = (hog_imagem - hog_imagem.min()) / (hog_imagem.max() - hog_imagem.min() + 1e-9)
    hog_imagem_uint8 = (hog_imagem_norm * 255).astype(np.uint8)
    
    return hog_vector, hog_imagem_uint8


def extrair_hog_completo(imagem, orientacoes=9, pixels_por_celula=(8, 8), celulas_por_bloco=(2, 2)):
    """
    Extrai características HOG completas com estatísticas adicionais.
    
    Args:
        imagem: Imagem de entrada (numpy.ndarray)
        orientacoes: Número de orientações (int, default=9)
        pixels_por_celula: Tamanho da célula em pixels (tuple, default=(8,8))
        celulas_por_bloco: Número de células por bloco (tuple, default=(2,2))
    
    Returns:
        dict: Dicionário com características HOG completas
    """
    try:
        # Extrair HOG básico
        hog_vector, hog_imagem = extrair_hog_basico(
            imagem, orientacoes, pixels_por_celula, celulas_por_bloco
        )
        
        # Calcular estatísticas
        hog_mean = float(np.mean(hog_vector))
        hog_std = float(np.std(hog_vector))
        hog_min = float(np.min(hog_vector))
        hog_max = float(np.max(hog_vector))
        hog_median = float(np.median(hog_vector))
        
        # Calcular dimensões
        dimensoes = len(hog_vector)
        
        # Calcular energia (soma dos quadrados)
        energia = float(np.sum(hog_vector ** 2))
        
        # Calcular entropia
        hist, _ = np.histogram(hog_vector, bins=50, density=True)
        hist = hist[hist > 0]  # Remover zeros
        entropia = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        return {
            'hog_vector': hog_vector.tolist(),
            'hog_imagem': hog_imagem,
            'hog_mean': hog_mean,
            'hog_std': hog_std,
            'hog_min': hog_min,
            'hog_max': hog_max,
            'hog_median': hog_median,
            'hog_dimensions': dimensoes,
            'hog_energy': energia,
            'hog_entropy': entropia,
            'orientacoes': orientacoes,
            'pixels_por_celula': pixels_por_celula,
            'celulas_por_bloco': celulas_por_bloco,
            'extraction_success': True
        }
        
    except Exception as e:
        return {
            'hog_vector': [],
            'hog_imagem': None,
            'hog_mean': 0.0,
            'hog_std': 0.0,
            'hog_min': 0.0,
            'hog_max': 0.0,
            'hog_median': 0.0,
            'hog_dimensions': 0,
            'hog_energy': 0.0,
            'hog_entropy': 0.0,
            'orientacoes': orientacoes,
            'pixels_por_celula': pixels_por_celula,
            'celulas_por_bloco': celulas_por_bloco,
            'extraction_success': False,
            'error': str(e)
        }


def extrair_hog_multiplas_imagens(imagens, orientacoes=9, pixels_por_celula=(8, 8), celulas_por_bloco=(2, 2)):
    """
    Extrai características HOG de múltiplas imagens.
    
    Args:
        imagens: Lista de imagens (list)
        orientacoes: Número de orientações (int, default=9)
        pixels_por_celula: Tamanho da célula em pixels (tuple, default=(8,8))
        celulas_por_bloco: Número de células por bloco (tuple, default=(2,2))
    
    Returns:
        list: Lista de dicionários com características HOG de cada imagem
    """
    resultados = []
    
    for i, imagem in enumerate(imagens):
        resultado = extrair_hog_completo(imagem, orientacoes, pixels_por_celula, celulas_por_bloco)
        resultado['imagem_id'] = i
        resultados.append(resultado)
    
    return resultados


def comparar_hog_imagens(hog1, hog2, metodo='cosine'):
    """
    Compara duas imagens baseado em suas características HOG.
    
    Args:
        hog1: Características HOG da primeira imagem (dict)
        hog2: Características HOG da segunda imagem (dict)
        metodo: Método de comparação ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        float: Similaridade entre as imagens (0-1)
    """
    if not hog1.get('extraction_success') or not hog2.get('extraction_success'):
        return 0.0
    
    vector1 = np.array(hog1['hog_vector'])
    vector2 = np.array(hog2['hog_vector'])
    
    if len(vector1) != len(vector2):
        return 0.0
    
    if metodo == 'cosine':
        # Similaridade do cosseno
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similaridade = dot_product / (norm1 * norm2)
        return float(max(0.0, similaridade))
    
    elif metodo == 'euclidean':
        # Distância euclidiana normalizada
        distancia = np.linalg.norm(vector1 - vector2)
        max_distancia = np.sqrt(len(vector1))  # Distância máxima possível
        similaridade = 1.0 - (distancia / max_distancia)
        return float(max(0.0, similaridade))
    
    elif metodo == 'manhattan':
        # Distância de Manhattan normalizada
        distancia = np.sum(np.abs(vector1 - vector2))
        max_distancia = np.sum(np.abs(vector1)) + np.sum(np.abs(vector2))
        similaridade = 1.0 - (distancia / max_distancia)
        return float(max(0.0, similaridade))
    
    else:
        raise ValueError(f"Método de comparação '{metodo}' não suportado")


def analisar_forma_hog(caracteristicas_hog):
    """
    Analisa a forma do objeto baseado nas características HOG.
    
    Args:
        caracteristicas_hog: Dicionário com características HOG (dict)
    
    Returns:
        str: Descrição da forma baseada em HOG
    """
    if not caracteristicas_hog.get('extraction_success'):
        return "Análise HOG falhou"
    
    energia = caracteristicas_hog['hog_energy']
    entropia = caracteristicas_hog['hog_entropy']
    std = caracteristicas_hog['hog_std']
    
    # Análise baseada na energia
    if energia > 1000:
        forma_energia = "muito definida"
    elif energia > 500:
        forma_energia = "bem definida"
    elif energia > 100:
        forma_energia = "moderadamente definida"
    else:
        forma_energia = "pouco definida"
    
    # Análise baseada na entropia (complexidade)
    if entropia > 8:
        forma_complexidade = "muito complexa"
    elif entropia > 6:
        forma_complexidade = "complexa"
    elif entropia > 4:
        forma_complexidade = "moderadamente complexa"
    else:
        forma_complexidade = "simples"
    
    # Análise baseada no desvio padrão (variação)
    if std > 0.1:
        forma_variacao = "com muita variação"
    elif std > 0.05:
        forma_variacao = "com variação moderada"
    else:
        forma_variacao = "com pouca variação"
    
    return f"Forma {forma_energia}, {forma_complexidade} e {forma_variacao}"


def visualizar_hog(imagem, orientacoes=9, pixels_por_celula=(8, 8), celulas_por_bloco=(2, 2)):
    """
    Visualiza as características HOG de uma imagem.
    
    Args:
        imagem: Imagem de entrada (numpy.ndarray)
        orientacoes: Número de orientações (int, default=9)
        pixels_por_celula: Tamanho da célula em pixels (tuple, default=(8,8))
        celulas_por_bloco: Número de células por bloco (tuple, default=(2,2))
    
    Returns:
        numpy.ndarray: Imagem de visualização HOG
    """
    _, hog_imagem = extrair_hog_basico(imagem, orientacoes, pixels_por_celula, celulas_por_bloco)
    return hog_imagem


def calcular_dimensoes_hog(altura, largura, pixels_por_celula=(8, 8), celulas_por_bloco=(2, 2), orientacoes=9):
    """
    Calcula as dimensões esperadas do vetor HOG.
    
    Args:
        altura: Altura da imagem (int)
        largura: Largura da imagem (int)
        pixels_por_celula: Tamanho da célula em pixels (tuple, default=(8,8))
        celulas_por_bloco: Número de células por bloco (tuple, default=(2,2))
        orientacoes: Número de orientações (int, default=9)
    
    Returns:
        int: Dimensões esperadas do vetor HOG
    """
    # Calcular número de células
    celulas_y = altura // pixels_por_celula[1]
    celulas_x = largura // pixels_por_celula[0]
    
    # Calcular número de blocos
    blocos_y = celulas_y - celulas_por_bloco[1] + 1
    blocos_x = celulas_x - celulas_por_bloco[0] + 1
    
    # Calcular dimensões totais
    dimensoes = blocos_y * blocos_x * celulas_por_bloco[0] * celulas_por_bloco[1] * orientacoes
    
    return dimensoes

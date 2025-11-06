import cv2
import numpy as np


def calcular_gradientes(imagem):
    """
    Calcula os gradientes da imagem usando filtros Sobel.
    
    Args:
        imagem: Imagem em escala de cinza
        
    Returns:
        tuple: (magnitude, angulo) dos gradientes
    """
    grad_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angulo = np.arctan2(grad_y, grad_x)
    angulo = np.degrees(angulo) % 180
    
    return magnitude, angulo


def calcular_hog_array(imagem, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Calcula o Histogram of Oriented Gradients (HOG) de uma imagem.
    
    Args:
        imagem: Imagem em escala de cinza
        orientations: Número de orientações do gradiente (padrão: 9)
        pixels_per_cell: Tamanho da célula em pixels (padrão: (8, 8))
        cells_per_block: Número de células por bloco (padrão: (2, 2))
        
    Returns:
        list: Vetor HOG normalizado
    """
    altura, largura = imagem.shape
    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block

    magnitude, angulo = calcular_gradientes(imagem)
    n_cells_h = altura // cell_h
    n_cells_w = largura // cell_w
    cell_histograms = np.zeros((n_cells_h, n_cells_w, orientations))
    bin_size = 180 / orientations

    for i in range(n_cells_h):
        for j in range(n_cells_w):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ang = angulo[y_start:y_end, x_start:x_end]
            for y in range(cell_h):
                for x in range(cell_w):
                    mag_val = cell_mag[y, x]
                    ang_val = cell_ang[y, x]
                    bin_idx = ang_val / bin_size
                    bin_low = int(bin_idx) % orientations
                    bin_high = (bin_low + 1) % orientations
                    weight = bin_idx - int(bin_idx)
                    cell_histograms[i, j, bin_low] += mag_val * (1 - weight)
                    cell_histograms[i, j, bin_high] += mag_val * weight

    n_blocks_h = n_cells_h - block_h + 1
    n_blocks_w = n_cells_w - block_w + 1
    hog_features = []

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = cell_histograms[i:i+block_h, j:j+block_w, :]
            block_vector = block.flatten()
            norm = np.sqrt(np.sum(block_vector**2) + 1e-6)
            block_vector = block_vector / norm
            hog_features.extend(block_vector)

    hog_features = np.array(hog_features, dtype=np.float32)
    norm_global = np.linalg.norm(hog_features)
    hog_features_normalized = hog_features / (norm_global + 1e-8)

    return hog_features_normalized.tolist()


def extrair_hog(caminho_imagem, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extrai características HOG de uma imagem a partir do caminho do arquivo.
    
    Args:
        caminho_imagem: Caminho para o arquivo de imagem
        orientations: Número de orientações do gradiente (padrão: 9)
        pixels_per_cell: Tamanho da célula em pixels (padrão: (8, 8))
        cells_per_block: Número de células por bloco (padrão: (2, 2))
        
    Returns:
        list: Vetor HOG normalizado
        
    Raises:
        ValueError: Se a imagem não puder ser carregada
    """
    # Carregar imagem
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")
    
    # Converter para escala de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Calcular HOG
    return calcular_hog_array(cinza, orientations, pixels_per_cell, cells_per_block)


def extrair_hog_de_imagem(imagem_bgr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extrai características HOG de uma imagem já carregada (BGR).
    
    Args:
        imagem_bgr: Imagem BGR já carregada (np.ndarray)
        orientations: Número de orientações do gradiente (padrão: 9)
        pixels_per_cell: Tamanho da célula em pixels (padrão: (8, 8))
        cells_per_block: Número de células por bloco (padrão: (2, 2))
        
    Returns:
        list: Vetor HOG normalizado
        
    Raises:
        ValueError: Se a imagem for inválida
    """
    if imagem_bgr is None:
        raise ValueError("Imagem inválida (None)")
    
    # Converter para escala de cinza se necessário
    if len(imagem_bgr.shape) == 3:
        cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem_bgr
    
    # Calcular HOG
    return calcular_hog_array(cinza, orientations, pixels_per_cell, cells_per_block)


def extrair_hog_lote(lista_caminhos, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extrai características HOG para uma lista de imagens.
    
    Args:
        lista_caminhos: Lista de caminhos para as imagens
        orientations: Número de orientações do gradiente (padrão: 9)
        pixels_per_cell: Tamanho da célula em pixels (padrão: (8, 8))
        cells_per_block: Número de células por bloco (padrão: (2, 2))
        
    Returns:
        list: Lista de vetores HOG (uma lista por imagem)
    """
    resultados = []
    
    for caminho in lista_caminhos:
        try:
            hog_features = extrair_hog(caminho, orientations, pixels_per_cell, cells_per_block)
            resultados.append(hog_features)
        except ValueError as e:
            print(f"Erro ao processar {caminho}: {e}")
            resultados.append(None)
    
    return resultados
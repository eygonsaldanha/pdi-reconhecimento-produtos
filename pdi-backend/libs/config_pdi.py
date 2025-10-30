#!/usr/bin/env python3
"""
Configurações centralizadas para o pipeline de PDI.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ConfigPreprocessing:
    """Configurações de pré-processamento."""
    gaussian_kernel: Tuple[int, int] = (5, 5)
    canny_low: int = 100
    canny_high: int = 200
    gaussian_sigma: float = 0.0  # Automático


@dataclass
class ConfigSegmentation:
    """Configurações de segmentação."""
    min_area: int = 1000
    border_margin: int = 1
    flood_fill_seeds: Optional[List[Tuple[int, int]]] = None
    
    def __post_init__(self):
        if self.flood_fill_seeds is None:
            self.flood_fill_seeds = [(0, 0), (0, 0), (0, 0), (0, 0)]  # Será calculado dinamicamente


@dataclass
class ConfigFeatures:
    """Configurações de extração de características."""
    # LBP
    lbp_p: int = 8
    lbp_r: int = 1
    lbp_method: str = "uniform"
    
    # GLCM
    glcm_distances: Optional[List[int]] = None
    glcm_angles: Optional[List[float]] = None
    glcm_levels: int = 256
    
    # HOG
    hog_orientations: int = 9
    hog_pixels_per_cell: Tuple[int, int] = (8, 8)
    hog_cells_per_block: Tuple[int, int] = (2, 2)
    hog_block_norm: str = "L2-Hys"
    
    def __post_init__(self):
        if self.glcm_distances is None:
            self.glcm_distances = [1, 2, 3]
        if self.glcm_angles is None:
            import numpy as np
            self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]


@dataclass
class ConfigVisualization:
    """Configurações de visualização."""
    save_images: bool = True
    show_plots: bool = False
    output_dir: str = "processed_images"
    image_format: str = "png"
    dpi: int = 300
    figsize: Tuple[int, int] = (15, 10)


@dataclass
class ConfigPipeline:
    """Configuração completa do pipeline."""
    preprocessing: ConfigPreprocessing
    segmentation: ConfigSegmentation
    features: ConfigFeatures
    visualization: ConfigVisualization
    
    def __init__(self, **kwargs):
        # Configurações padrão
        self.preprocessing = ConfigPreprocessing()
        self.segmentation = ConfigSegmentation()
        self.features = ConfigFeatures()
        self.visualization = ConfigVisualization()
        
        # Aplicar configurações personalizadas
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    # Atualizar configuração específica
                    config_obj = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)


# Configurações pré-definidas
CONFIGURACOES_PREDEFINIDAS = {
    'rapido': {
        'preprocessing': {
            'gaussian_kernel': (3, 3),
            'canny_low': 50,
            'canny_high': 150
        },
        'segmentation': {
            'min_area': 500
        },
        'features': {
            'lbp_p': 8,
            'lbp_r': 1,
            'hog_orientations': 9,
            'hog_pixels_per_cell': (16, 16)
        },
        'visualization': {
            'show_plots': False,
            'save_images': False
        }
    },
    
    'balanceado': {
        'preprocessing': {
            'gaussian_kernel': (5, 5),
            'canny_low': 100,
            'canny_high': 200
        },
        'segmentation': {
            'min_area': 1000
        },
        'features': {
            'lbp_p': 8,
            'lbp_r': 1,
            'hog_orientations': 9,
            'hog_pixels_per_cell': (8, 8)
        },
        'visualization': {
            'show_plots': True,
            'save_images': True
        }
    },
    
    'detalhado': {
        'preprocessing': {
            'gaussian_kernel': (7, 7),
            'canny_low': 80,
            'canny_high': 220
        },
        'segmentation': {
            'min_area': 2000
        },
        'features': {
            'lbp_p': 16,
            'lbp_r': 2,
            'glcm_distances': [1, 2, 3, 4],
            'glcm_angles': [0, 30, 60, 90, 120, 150],
            'hog_orientations': 18,
            'hog_pixels_per_cell': (4, 4)
        },
        'visualization': {
            'show_plots': True,
            'save_images': True,
            'dpi': 600
        }
    }
}


def obter_configuracao(tipo: str = 'balanceado') -> ConfigPipeline:
    """
    Obtém uma configuração pré-definida.
    
    Args:
        tipo: Tipo de configuração ('rapido', 'balanceado', 'detalhado')
        
    Returns:
        ConfigPipeline: Configuração do pipeline
    """
    if tipo not in CONFIGURACOES_PREDEFINIDAS:
        raise ValueError(f"Tipo de configuração '{tipo}' não encontrado. "
                        f"Tipos disponíveis: {list(CONFIGURACOES_PREDEFINIDAS.keys())}")
    
    return ConfigPipeline(**CONFIGURACOES_PREDEFINIDAS[tipo])


def criar_configuracao_personalizada(**kwargs) -> ConfigPipeline:
    """
    Cria uma configuração personalizada.
    
    Args:
        **kwargs: Configurações personalizadas
        
    Returns:
        ConfigPipeline: Configuração personalizada
    """
    return ConfigPipeline(**kwargs)


def validar_configuracao(config: ConfigPipeline) -> List[str]:
    """
    Valida uma configuração do pipeline.
    
    Args:
        config: Configuração a ser validada
        
    Returns:
        List[str]: Lista de erros encontrados (vazia se válida)
    """
    erros = []
    
    # Validar pré-processamento
    if config.preprocessing.gaussian_kernel[0] % 2 == 0 or config.preprocessing.gaussian_kernel[1] % 2 == 0:
        erros.append("Kernel gaussiano deve ter dimensões ímpares")
    
    if config.preprocessing.canny_low >= config.preprocessing.canny_high:
        erros.append("Limiar baixo do Canny deve ser menor que o limiar alto")
    
    # Validar segmentação
    if config.segmentation.min_area < 0:
        erros.append("Área mínima deve ser positiva")
    
    if config.segmentation.border_margin < 0:
        erros.append("Margem da borda deve ser não negativa")
    
    # Validar características
    if config.features.lbp_p <= 0:
        erros.append("LBP P deve ser positivo")
    
    if config.features.lbp_r <= 0:
        erros.append("LBP R deve ser positivo")
    
    if config.features.hog_orientations <= 0:
        erros.append("HOG orientations deve ser positivo")
    
    if (config.features.hog_pixels_per_cell[0] <= 0 or 
        config.features.hog_pixels_per_cell[1] <= 0):
        erros.append("HOG pixels_per_cell deve ser positivo")
    
    if (config.features.hog_cells_per_block[0] <= 0 or 
        config.features.hog_cells_per_block[1] <= 0):
        erros.append("HOG cells_per_block deve ser positivo")
    
    # Validar visualização
    if config.visualization.dpi <= 0:
        erros.append("DPI deve ser positivo")
    
    if config.visualization.figsize[0] <= 0 or config.visualization.figsize[1] <= 0:
        erros.append("Tamanho da figura deve ser positivo")
    
    return erros


def converter_para_dict(config: ConfigPipeline) -> Dict[str, Any]:
    """
    Converte uma configuração para dicionário.
    
    Args:
        config: Configuração a ser convertida
        
    Returns:
        Dict[str, Any]: Configuração em formato de dicionário
    """
    return {
        'preprocessing': {
            'gaussian_kernel': config.preprocessing.gaussian_kernel,
            'canny_low': config.preprocessing.canny_low,
            'canny_high': config.preprocessing.canny_high,
            'gaussian_sigma': config.preprocessing.gaussian_sigma
        },
        'segmentation': {
            'min_area': config.segmentation.min_area,
            'border_margin': config.segmentation.border_margin,
            'flood_fill_seeds': config.segmentation.flood_fill_seeds
        },
        'features': {
            'lbp_p': config.features.lbp_p,
            'lbp_r': config.features.lbp_r,
            'lbp_method': config.features.lbp_method,
            'glcm_distances': config.features.glcm_distances,
            'glcm_angles': config.features.glcm_angles,
            'glcm_levels': config.features.glcm_levels,
            'hog_orientations': config.features.hog_orientations,
            'hog_pixels_per_cell': config.features.hog_pixels_per_cell,
            'hog_cells_per_block': config.features.hog_cells_per_block,
            'hog_block_norm': config.features.hog_block_norm
        },
        'visualization': {
            'save_images': config.visualization.save_images,
            'show_plots': config.visualization.show_plots,
            'output_dir': config.visualization.output_dir,
            'image_format': config.visualization.image_format,
            'dpi': config.visualization.dpi,
            'figsize': config.visualization.figsize
        }
    }


def exemplo_uso_configuracao():
    """Exemplo de uso das configurações."""
    print("⚙️ EXEMPLO DE USO DAS CONFIGURAÇÕES")
    print("=" * 50)
    
    # Usar configuração pré-definida
    print("\n1. Configuração pré-definida 'rapido':")
    config_rapido = obter_configuracao('rapido')
    print(f"   Kernel gaussiano: {config_rapido.preprocessing.gaussian_kernel}")
    print(f"   Área mínima: {config_rapido.segmentation.min_area}")
    print(f"   HOG orientações: {config_rapido.features.hog_orientations}")
    
    # Validar configuração
    erros = validar_configuracao(config_rapido)
    if erros:
        print(f"   ❌ Erros encontrados: {erros}")
    else:
        print("   ✅ Configuração válida")
    
    # Criar configuração personalizada
    print("\n2. Configuração personalizada:")
    config_personalizada = criar_configuracao_personalizada(
        preprocessing={'gaussian_kernel': (9, 9)},
        segmentation={'min_area': 5000},
        features={'hog_orientations': 12}
    )
    print(f"   Kernel gaussiano: {config_personalizada.preprocessing.gaussian_kernel}")
    print(f"   Área mínima: {config_personalizada.segmentation.min_area}")
    print(f"   HOG orientações: {config_personalizada.features.hog_orientations}")
    
    # Converter para dicionário
    print("\n3. Configuração em formato dicionário:")
    config_dict = converter_para_dict(config_personalizada)
    print(f"   Chaves disponíveis: {list(config_dict.keys())}")


if __name__ == "__main__":
    exemplo_uso_configuracao()

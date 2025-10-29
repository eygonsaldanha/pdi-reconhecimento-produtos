#!/usr/bin/env python3
"""
Pipeline completo de PDI integrando todas as bibliotecas na ordem ideal.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

# Importar todas as bibliotecas de PDI
from preprocessing import (
    converter_para_cinza,
    aplicar_filtro_gaussiano,
    detectar_bordas_canny
)

from segmentation import (
    segmentar_objeto_com_flood_fill,
    encontrar_contornos,
    filtrar_contornos_borda,
    desenhar_contornos
)

from geometrização import (
    extrair_caracteristicas_geometricas_completas,
    filtrar_contornos_por_area,
    obter_maior_contorno
)

from texture import (
    extrair_lbp,
    extrair_glcm,
    extrair_caracteristicas_textura_completas
)

from form import (
    extrair_hog_completo,
    analisar_forma_hog
)

from visualization import (
    plotar_resultados_segmentacao,
    plotar_caracteristicas,
    plotar_histograma_lbp
)


class PipelinePDI:
    """
    Pipeline completo de Processamento Digital de Imagens.
    
    Ordem ideal de execução:
    1. Pré-processamento
    2. Segmentação
    3. Extração de características (Geometria → Textura → Forma)
    4. Visualização
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o pipeline com configurações opcionais.
        
        Args:
            config: Dicionário com configurações do pipeline
        """
        self.config = config or self._get_default_config()
        self.timings = {}
        self.resultados = {}
    
    def _get_default_config(self) -> Dict:
        """Retorna configurações padrão do pipeline."""
        return {
            'preprocessing': {
                'gaussian_kernel': (5, 5),
                'canny_low': 100,
                'canny_high': 200
            },
            'segmentation': {
                'min_area': 1000,
                'border_margin': 1
            },
            'features': {
                'lbp_params': {'P': 8, 'R': 1},
                'glcm_params': {
                    'distances': [1, 2, 3],
                    'angles': [0, 45, 90, 135]
                },
                'hog_params': {
                    'orientacoes': 9,
                    'pixels_por_celula': (8, 8),
                    'celulas_por_bloco': (2, 2)
                }
            },
            'visualization': {
                'save_images': True,
                'show_plots': False
            }
        }
    
    def processar_imagem(self, caminho_imagem: str) -> Dict:
        """
        Processa uma imagem completa seguindo a ordem ideal.
        
        Args:
            caminho_imagem: Caminho para a imagem
            
        Returns:
            dict: Resultados completos do processamento
        """
        inicio_total = time.time()
        
        try:
            # FASE 1: CARREGAMENTO E PRÉ-PROCESSAMENTO
            print("🔄 FASE 1: Carregamento e Pré-processamento")
            inicio = time.time()
            
            imagem_original = cv2.imread(caminho_imagem)
            if imagem_original is None:
                raise ValueError(f"Falha ao carregar a imagem: {caminho_imagem}")
            
            # Pré-processamento
            cinza = converter_para_cinza(imagem_original)
            gaussiano = aplicar_filtro_gaussiano(
                cinza, 
                self.config['preprocessing']['gaussian_kernel']
            )
            bordas = detectar_bordas_canny(
                gaussiano,
                self.config['preprocessing']['canny_low'],
                self.config['preprocessing']['canny_high']
            )
            
            self.timings['preprocessing'] = time.time() - inicio
            print(f"✅ Pré-processamento concluído em {self.timings['preprocessing']:.3f}s")
            
            # FASE 2: SEGMENTAÇÃO
            print("\n🔄 FASE 2: Segmentação")
            inicio = time.time()
            
            mascara = segmentar_objeto_com_flood_fill(gaussiano)
            contornos_todos = encontrar_contornos(mascara)
            contornos_filtrados = filtrar_contornos_borda(
                contornos_todos,
                gaussiano.shape[1],
                gaussiano.shape[0],
                self.config['segmentation']['border_margin']
            )
            
            # Filtrar por área mínima
            contornos_finais = filtrar_contornos_por_area(
                contornos_filtrados,
                self.config['segmentation']['min_area']
            )
            
            # Desenhar contornos
            imagem_contornos = desenhar_contornos(imagem_original, contornos_finais)
            
            self.timings['segmentation'] = time.time() - inicio
            print(f"✅ Segmentação concluída em {self.timings['segmentation']:.3f}s")
            print(f"   Contornos encontrados: {len(contornos_finais)}")
            
            # FASE 3: EXTRAÇÃO DE CARACTERÍSTICAS
            print("\n🔄 FASE 3: Extração de Características")
            inicio = time.time()
            
            caracteristicas = {}
            
            # 3.1 Características Geométricas (mais rápidas)
            print("   📐 Extraindo características geométricas...")
            if contornos_finais:
                maior_contorno = obter_maior_contorno(contornos_finais)
                caracteristicas['geometricas'] = extrair_caracteristicas_geometricas_completas(maior_contorno)
            else:
                caracteristicas['geometricas'] = {
                    'area': 0, 'perimetro': 0, 'circularidade': 0,
                    'aspect_ratio': 0, 'bounding_box': (0, 0, 0, 0), 'centroide': (0, 0)
                }
            
            # 3.2 Características de Textura (paralelo)
            print("   🎨 Extraindo características de textura...")
            lbp_img, lbp_hist = extrair_lbp(cinza, **self.config['features']['lbp_params'])
            glcm_caracteristicas = extrair_glcm(cinza, **self.config['features']['glcm_params'])
            
            caracteristicas['textura'] = {
                'lbp': {
                    'imagem': lbp_img,
                    'histograma': lbp_hist.tolist(),
                    'media': float(np.mean(lbp_hist)),
                    'desvio': float(np.std(lbp_hist))
                },
                'glcm': glcm_caracteristicas
            }
            
            # 3.3 Características de Forma (mais informativas)
            print("   🔺 Extraindo características de forma (HOG)...")
            hog_caracteristicas = extrair_hog_completo(
                imagem_original,
                **self.config['features']['hog_params']
            )
            caracteristicas['forma'] = hog_caracteristicas
            
            self.timings['features'] = time.time() - inicio
            print(f"✅ Extração de características concluída em {self.timings['features']:.3f}s")
            
            # FASE 4: ANÁLISE E RESULTADOS
            print("\n🔄 FASE 4: Análise e Resultados")
            inicio = time.time()
            
            # Análise de qualidade
            qualidade = self._analisar_qualidade(caracteristicas, contornos_finais)
            
            # Análise de forma
            if caracteristicas['forma']['extraction_success']:
                forma_analise = analisar_forma_hog(caracteristicas['forma'])
            else:
                forma_analise = "Análise de forma não disponível"
            
            # Preparar resultados
            self.resultados = {
                'imagem_original': imagem_original,
                'preprocessamento': {
                    'cinza': cinza,
                    'gaussiano': gaussiano,
                    'bordas': bordas
                },
                'segmentacao': {
                    'mascara': mascara,
                    'contornos_todos': contornos_todos,
                    'contornos_filtrados': contornos_filtrados,
                    'contornos_finais': contornos_finais,
                    'imagem_contornos': imagem_contornos
                },
                'caracteristicas': caracteristicas,
                'analise': {
                    'qualidade': qualidade,
                    'forma': forma_analise
                },
                'timings': self.timings,
                'success': True
            }
            
            self.timings['analysis'] = time.time() - inicio
            self.timings['total'] = time.time() - inicio_total
            
            print(f"✅ Análise concluída em {self.timings['analysis']:.3f}s")
            print(f"🎉 Pipeline completo executado em {self.timings['total']:.3f}s")
            
            return self.resultados
            
        except Exception as e:
            print(f"❌ Erro no pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'timings': self.timings
            }
    
    def _analisar_qualidade(self, caracteristicas: Dict, contornos: List) -> Dict:
        """
        Analisa a qualidade do processamento.
        
        Args:
            caracteristicas: Características extraídas
            contornos: Lista de contornos
            
        Returns:
            dict: Análise de qualidade
        """
        qualidade = {
            'score_geral': 0.0,
            'detalhes': {}
        }
        
        # Score baseado no número de contornos
        if len(contornos) > 0:
            qualidade['detalhes']['contornos'] = 'Bom'
            score_contornos = 0.8
        else:
            qualidade['detalhes']['contornos'] = 'Ruim'
            score_contornos = 0.0
        
        # Score baseado na área do objeto
        area = caracteristicas['geometricas']['area']
        if area > 10000:
            qualidade['detalhes']['area'] = 'Excelente'
            score_area = 1.0
        elif area > 5000:
            qualidade['detalhes']['area'] = 'Bom'
            score_area = 0.8
        elif area > 1000:
            qualidade['detalhes']['area'] = 'Regular'
            score_area = 0.6
        else:
            qualidade['detalhes']['area'] = 'Ruim'
            score_area = 0.3
        
        # Score baseado na circularidade
        circularidade = caracteristicas['geometricas']['circularidade']
        if circularidade > 0.7:
            qualidade['detalhes']['forma'] = 'Muito definida'
            score_forma = 1.0
        elif circularidade > 0.4:
            qualidade['detalhes']['forma'] = 'Bem definida'
            score_forma = 0.8
        else:
            qualidade['detalhes']['forma'] = 'Pouco definida'
            score_forma = 0.5
        
        # Score baseado na extração HOG
        if caracteristicas['forma']['extraction_success']:
            qualidade['detalhes']['hog'] = 'Sucesso'
            score_hog = 1.0
        else:
            qualidade['detalhes']['hog'] = 'Falha'
            score_hog = 0.0
        
        # Score geral (média ponderada)
        qualidade['score_geral'] = (
            score_contornos * 0.2 +
            score_area * 0.3 +
            score_forma * 0.3 +
            score_hog * 0.2
        )
        
        return qualidade
    
    def visualizar_resultados(self, salvar_imagens: bool = True):
        """
        Visualiza os resultados do processamento.
        
        Args:
            salvar_imagens: Se deve salvar imagens intermediárias
        """
        if not self.resultados.get('success', False):
            print("❌ Nenhum resultado disponível para visualização")
            return
        
        print("\n🖼️ VISUALIZANDO RESULTADOS:")
        print("-" * 40)
        
        # Plotar resultados de segmentação
        try:
            plotar_resultados_segmentacao(self.resultados)
            print("✅ Gráfico de segmentação gerado")
        except Exception as e:
            print(f"⚠️ Erro na visualização de segmentação: {e}")
        
        # Plotar características
        try:
            plotar_caracteristicas(self.resultados)
            print("✅ Gráfico de características gerado")
        except Exception as e:
            print(f"⚠️ Erro na visualização de características: {e}")
        
        # Plotar histograma LBP
        try:
            plotar_histograma_lbp(self.resultados)
            print("✅ Histograma LBP gerado")
        except Exception as e:
            print(f"⚠️ Erro na visualização do histograma LBP: {e}")
        
        # Salvar imagens se solicitado
        if salvar_imagens:
            self._salvar_imagens()
    
    def _salvar_imagens(self):
        """Salva imagens intermediárias do processamento."""
        try:
            import os
            pasta_saida = "processed_images"
            os.makedirs(pasta_saida, exist_ok=True)
            
            # Salvar imagens principais
            cv2.imwrite(f"{pasta_saida}/original.png", self.resultados['imagem_original'])
            cv2.imwrite(f"{pasta_saida}/cinza.png", self.resultados['preprocessamento']['cinza'])
            cv2.imwrite(f"{pasta_saida}/gaussiano.png", self.resultados['preprocessamento']['gaussiano'])
            cv2.imwrite(f"{pasta_saida}/bordas.png", self.resultados['preprocessamento']['bordas'])
            cv2.imwrite(f"{pasta_saida}/mascara.png", self.resultados['segmentacao']['mascara'])
            cv2.imwrite(f"{pasta_saida}/contornos.png", self.resultados['segmentacao']['imagem_contornos'])
            
            # Salvar características visuais
            if 'textura' in self.resultados['caracteristicas']:
                lbp_img = self.resultados['caracteristicas']['textura']['lbp']['imagem']
                cv2.imwrite(f"{pasta_saida}/lbp.png", lbp_img)
            
            print(f"✅ Imagens salvas em: {pasta_saida}/")
            
        except Exception as e:
            print(f"⚠️ Erro ao salvar imagens: {e}")
    
    def obter_resumo(self) -> Dict:
        """
        Retorna um resumo dos resultados do processamento.
        
        Returns:
            dict: Resumo dos resultados
        """
        if not self.resultados.get('success', False):
            return {'error': 'Nenhum resultado disponível'}
        
        caracteristicas = self.resultados['caracteristicas']
        
        return {
            'imagem': {
                'dimensoes': self.resultados['imagem_original'].shape,
                'tipo': 'BGR' if len(self.resultados['imagem_original'].shape) == 3 else 'GRAY'
            },
            'segmentacao': {
                'contornos_encontrados': len(self.resultados['segmentacao']['contornos_finais']),
                'area_objeto': caracteristicas['geometricas']['area'],
                'perimetro_objeto': caracteristicas['geometricas']['perimetro']
            },
            'caracteristicas': {
                'geometricas': {
                    'circularidade': caracteristicas['geometricas']['circularidade'],
                    'aspect_ratio': caracteristicas['geometricas']['aspect_ratio']
                },
                'textura': {
                    'lbp_bins': len(caracteristicas['textura']['lbp']['histograma']),
                    'glcm_propriedades': len(caracteristicas['textura']['glcm'])
                },
                'forma': {
                    'hog_dimensoes': caracteristicas['forma'].get('hog_dimensions', 0),
                    'hog_energia': caracteristicas['forma'].get('hog_energy', 0)
                }
            },
            'qualidade': self.resultados['analise']['qualidade'],
            'performance': self.timings
        }


def exemplo_uso():
    """Exemplo de uso do pipeline completo."""
    print("🚀 EXEMPLO DE USO DO PIPELINE COMPLETO DE PDI")
    print("=" * 60)
    
    # Configuração personalizada
    config = {
        'preprocessing': {
            'gaussian_kernel': (5, 5),
            'canny_low': 100,
            'canny_high': 200
        },
        'segmentation': {
            'min_area': 1000,
            'border_margin': 1
        },
        'features': {
            'lbp_params': {'P': 8, 'R': 1},
            'glcm_params': {
                'distances': [1, 2, 3],
                'angles': [0, 45, 90, 135]
            },
            'hog_params': {
                'orientacoes': 9,
                'pixels_por_celula': (8, 8),
                'celulas_por_bloco': (2, 2)
            }
        }
    }
    
    # Criar pipeline
    pipeline = PipelinePDI(config)
    
    # Processar imagem
    try:
        resultados = pipeline.processar_imagem("apple.jpg")
        
        if resultados['success']:
            # Mostrar resumo
            resumo = pipeline.obter_resumo()
            print("\n📊 RESUMO DOS RESULTADOS:")
            print(f"   Dimensões: {resumo['imagem']['dimensoes']}")
            print(f"   Contornos: {resumo['segmentacao']['contornos_encontrados']}")
            print(f"   Área: {resumo['segmentacao']['area_objeto']:.2f} pixels²")
            print(f"   Circularidade: {resumo['caracteristicas']['geometricas']['circularidade']:.3f}")
            print(f"   HOG: {resumo['caracteristicas']['forma']['hog_dimensoes']} dimensões")
            print(f"   Qualidade: {resumo['qualidade']['score_geral']:.2f}")
            print(f"   Tempo total: {resumo['performance']['total']:.3f}s")
            
            # Visualizar resultados
            pipeline.visualizar_resultados()
            
        else:
            print(f"❌ Falha no processamento: {resultados.get('error', 'Erro desconhecido')}")
            
    except FileNotFoundError:
        print("❌ Arquivo de imagem não encontrado. Use uma imagem válida.")
    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    exemplo_uso()

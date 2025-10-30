#!/usr/bin/env python3
"""
Teste completo do pipeline de PDI integrado.
"""

import cv2
import numpy as np
import time
from pipeline_completo import PipelinePDI
from config_pdi import obter_configuracao, validar_configuracao, converter_para_dict


def criar_imagens_teste():
    """Cria imagens de teste com diferentes formas."""
    imagens = {}
    
    # Imagem 1: Círculo
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img1, (100, 100), 80, (255, 255, 255), -1)
    imagens['circulo'] = img1
    
    # Imagem 2: Retângulo
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img2, (50, 50), (150, 150), (255, 255, 255), -1)
    imagens['retangulo'] = img2
    
    # Imagem 3: Triângulo
    img3 = np.zeros((200, 200, 3), dtype=np.uint8)
    pts = np.array([[100, 30], [30, 170], [170, 170]], np.int32)
    cv2.fillPoly(img3, [pts], (255, 255, 255))
    imagens['triangulo'] = img3
    
    # Imagem 4: Elipse
    img4 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.ellipse(img4, (100, 100), (80, 40), 0, 0, 360, (255, 255, 255), -1)
    imagens['elipse'] = img4
    
    # Imagem 5: Forma complexa (múltiplos objetos)
    img5 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img5, (80, 80), 30, (255, 255, 255), -1)
    cv2.rectangle(img5, (120, 120), (180, 180), (255, 255, 255), -1)
    imagens['complexa'] = img5
    
    return imagens


def testar_configuracoes():
    """Testa diferentes configurações do pipeline."""
    print("⚙️ TESTANDO DIFERENTES CONFIGURAÇÕES")
    print("=" * 50)
    
    configuracoes = ['rapido', 'balanceado', 'detalhado']
    imagens_teste = criar_imagens_teste()
    
    for config_name in configuracoes:
        print(f"\n🔧 Testando configuração '{config_name}':")
        
        # Obter configuração
        config = obter_configuracao(config_name)
        
        # Validar configuração
        erros = validar_configuracao(config)
        if erros:
            print(f"   ❌ Configuração inválida: {erros}")
            continue
        
        # Criar pipeline
        pipeline = PipelinePDI(converter_para_dict(config))
        
        # Testar com imagem simples (círculo)
        print("   🧪 Testando com imagem círculo...")
        inicio = time.time()
        
        # Salvar imagem temporária
        cv2.imwrite("temp_circulo.jpg", imagens_teste['circulo'])
        
        try:
            resultados = pipeline.processar_imagem("temp_circulo.jpg")
            
            if resultados['success']:
                tempo = time.time() - inicio
                resumo = pipeline.obter_resumo()
                
                print(f"   ✅ Sucesso em {tempo:.3f}s")
                print(f"      Contornos: {resumo['segmentacao']['contornos_encontrados']}")
                print(f"      Área: {resumo['segmentacao']['area_objeto']:.0f} pixels²")
                print(f"      Circularidade: {resumo['caracteristicas']['geometricas']['circularidade']:.3f}")
                print(f"      Qualidade: {resumo['qualidade']['score_geral']:.2f}")
            else:
                print(f"   ❌ Falha: {resultados.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        finally:
            # Limpar arquivo temporário
            import os
            if os.path.exists("temp_circulo.jpg"):
                os.remove("temp_circulo.jpg")


def testar_pipeline_completo():
    """Testa o pipeline completo com diferentes imagens."""
    print("\n🚀 TESTANDO PIPELINE COMPLETO")
    print("=" * 50)
    
    # Usar configuração balanceada
    config = obter_configuracao('balanceado')
    pipeline = PipelinePDI(converter_para_dict(config))
    
    imagens_teste = criar_imagens_teste()
    resultados_teste = {}
    
    for nome, imagem in imagens_teste.items():
        print(f"\n📸 Processando {nome}:")
        
        # Salvar imagem temporária
        nome_arquivo = f"temp_{nome}.jpg"
        cv2.imwrite(nome_arquivo, imagem)
        
        try:
            inicio = time.time()
            resultados = pipeline.processar_imagem(nome_arquivo)
            tempo = time.time() - inicio
            
            if resultados['success']:
                resumo = pipeline.obter_resumo()
                resultados_teste[nome] = {
                    'sucesso': True,
                    'tempo': tempo,
                    'resumo': resumo
                }
                
                print(f"   ✅ Sucesso em {tempo:.3f}s")
                print(f"      Dimensões: {resumo['imagem']['dimensoes']}")
                print(f"      Contornos: {resumo['segmentacao']['contornos_encontrados']}")
                print(f"      Área: {resumo['segmentacao']['area_objeto']:.0f} pixels²")
                print(f"      Circularidade: {resumo['caracteristicas']['geometricas']['circularidade']:.3f}")
                print(f"      Aspect Ratio: {resumo['caracteristicas']['geometricas']['aspect_ratio']:.3f}")
                print(f"      HOG: {resumo['caracteristicas']['forma']['hog_dimensoes']} dimensões")
                print(f"      Qualidade: {resumo['qualidade']['score_geral']:.2f}")
            else:
                resultados_teste[nome] = {
                    'sucesso': False,
                    'erro': resultados.get('error', 'Erro desconhecido')
                }
                print(f"   ❌ Falha: {resultados.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            resultados_teste[nome] = {
                'sucesso': False,
                'erro': str(e)
            }
            print(f"   ❌ Erro: {e}")
        finally:
            # Limpar arquivo temporário
            import os
            if os.path.exists(nome_arquivo):
                os.remove(nome_arquivo)
    
    return resultados_teste


def analisar_resultados(resultados_teste):
    """Analisa os resultados dos testes."""
    print("\n📊 ANÁLISE DOS RESULTADOS")
    print("=" * 50)
    
    sucessos = sum(1 for r in resultados_teste.values() if r['sucesso'])
    total = len(resultados_teste)
    
    print(f"📈 Taxa de sucesso: {sucessos}/{total} ({sucessos/total*100:.1f}%)")
    
    if sucessos > 0:
        print("\n🎯 Análise por tipo de forma:")
        
        for nome, resultado in resultados_teste.items():
            if resultado['sucesso']:
                resumo = resultado['resumo']
                print(f"\n   {nome.upper()}:")
                print(f"      Tempo: {resultado['tempo']:.3f}s")
                print(f"      Contornos: {resumo['segmentacao']['contornos_encontrados']}")
                print(f"      Área: {resumo['segmentacao']['area_objeto']:.0f} pixels²")
                print(f"      Circularidade: {resumo['caracteristicas']['geometricas']['circularidade']:.3f}")
                print(f"      Qualidade: {resumo['qualidade']['score_geral']:.2f}")
                
                # Análise específica
                if nome == 'circulo':
                    circularidade = resumo['caracteristicas']['geometricas']['circularidade']
                    if circularidade > 0.8:
                        print("      ✅ Círculo bem detectado")
                    else:
                        print("      ⚠️ Círculo mal detectado")
                
                elif nome == 'retangulo':
                    aspect_ratio = resumo['caracteristicas']['geometricas']['aspect_ratio']
                    if 0.9 <= aspect_ratio <= 1.1:
                        print("      ✅ Retângulo quadrado bem detectado")
                    else:
                        print("      ⚠️ Retângulo não quadrado")
                
                elif nome == 'triangulo':
                    circularidade = resumo['caracteristicas']['geometricas']['circularidade']
                    if circularidade < 0.5:
                        print("      ✅ Triângulo bem detectado (baixa circularidade)")
                    else:
                        print("      ⚠️ Triângulo mal detectado")
    
    # Estatísticas de performance
    if sucessos > 0:
        tempos = [r['tempo'] for r in resultados_teste.values() if r['sucesso']]
        print(f"\n⏱️ Performance:")
        print(f"   Tempo médio: {np.mean(tempos):.3f}s")
        print(f"   Tempo mínimo: {np.min(tempos):.3f}s")
        print(f"   Tempo máximo: {np.max(tempos):.3f}s")


def testar_robustez():
    """Testa a robustez do pipeline com imagens problemáticas."""
    print("\n🛡️ TESTANDO ROBUSTEZ")
    print("=" * 50)
    
    config = obter_configuracao('balanceado')
    pipeline = PipelinePDI(converter_para_dict(config))
    
    # Teste 1: Imagem vazia
    print("\n1. Teste com imagem vazia:")
    img_vazia = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite("temp_vazia.jpg", img_vazia)
    
    try:
        resultados = pipeline.processar_imagem("temp_vazia.jpg")
        if resultados['success']:
            resumo = pipeline.obter_resumo()
            print(f"   ✅ Processou imagem vazia: {resumo['segmentacao']['contornos_encontrados']} contornos")
        else:
            print(f"   ⚠️ Falha esperada: {resultados.get('error', 'Erro desconhecido')}")
    except Exception as e:
        print(f"   ⚠️ Erro esperado: {e}")
    finally:
        import os
        if os.path.exists("temp_vazia.jpg"):
            os.remove("temp_vazia.jpg")
    
    # Teste 2: Imagem com ruído
    print("\n2. Teste com imagem ruidosa:")
    rng = np.random.default_rng()
    img_ruido = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite("temp_ruido.jpg", img_ruido)
    
    try:
        resultados = pipeline.processar_imagem("temp_ruido.jpg")
        if resultados['success']:
            resumo = pipeline.obter_resumo()
            print(f"   ✅ Processou imagem ruidosa: {resumo['segmentacao']['contornos_encontrados']} contornos")
        else:
            print(f"   ⚠️ Falha esperada: {resultados.get('error', 'Erro desconhecido')}")
    except Exception as e:
        print(f"   ⚠️ Erro esperado: {e}")
    finally:
        import os
        if os.path.exists("temp_ruido.jpg"):
            os.remove("temp_ruido.jpg")


def main():
    """Função principal do teste."""
    print("🧪 TESTE COMPLETO DO PIPELINE DE PDI")
    print("=" * 60)
    
    try:
        # Testar configurações
        testar_configuracoes()
        
        # Testar pipeline completo
        resultados_teste = testar_pipeline_completo()
        
        # Analisar resultados
        analisar_resultados(resultados_teste)
        
        # Testar robustez
        testar_robustez()
        
        print("\n🎉 TESTE COMPLETO FINALIZADO COM SUCESSO!")
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

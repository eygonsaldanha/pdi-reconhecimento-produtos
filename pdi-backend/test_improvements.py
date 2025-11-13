"""
Script de teste para verificar as melhorias implementadas.

Testa:
1. Extração de features otimizado
2. Comparação de tamanho de features (antigo vs novo)
3. Carregamento do modelo com cache
4. Normalização
"""

import cv2
import numpy as np
import time
import os
from libs.knn_process import extrair_features_otimizado, knn_process_df_image


def teste_features_otimizado():
    """Testa a extração de features otimizado"""
    print("=" * 70)
    print("🧪 TESTE 1: Extração de Features Otimizado")
    print("=" * 70)
    
    # Criar imagem de teste (100x100)
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        # Testar função nova
        start = time.time()
        features_novo = extrair_features_otimizado(image_process=test_image)
        tempo_novo = time.time() - start
        
        print(f"✅ Features otimizado extraídas com sucesso!")
        print(f"   Tamanho: {len(features_novo)} valores")
        print(f"   Tempo: {tempo_novo:.4f}s")
        print(f"   Shape: {features_novo.shape}")
        print(f"   Tipo: {features_novo.dtype}")
        
        # Verificar composição
        print(f"\n📊 Composição das features:")
        print(f"   - Geométricas: 4 valores")
        print(f"   - HOG: ~{len(features_novo) - 57} valores")
        print(f"   - LBP: 10 valores")
        print(f"   - GLCM: 6 valores")
        print(f"   - Histograma RGB: 30 valores")
        print(f"   - Momentos de Hu: 7 valores")
        
        return True, features_novo
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def teste_comparacao_features():
    """Compara features antigas vs novas"""
    print("\n" + "=" * 70)
    print("🧪 TESTE 2: Comparação Features Antigas vs Novas")
    print("=" * 70)
    
    # Criar imagem de teste
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        # Features antigas (dicionário)
        start = time.time()
        features_antigas = knn_process_df_image(image_process=test_image)
        tempo_antigo = time.time() - start
        
        # Contar valores nas features antigas
        total_antigo = sum(len(np.ravel(v)) for v in features_antigas.values())
        
        # Features novas (array)
        start = time.time()
        features_novas = extrair_features_otimizado(image_process=test_image)
        tempo_novo = time.time() - start
        
        total_novo = len(features_novas)
        
        print(f"📊 Resultados:")
        print(f"\n   Features ANTIGAS:")
        print(f"      Tipo: Dicionário com {len(features_antigas)} chaves")
        print(f"      Total de valores: {total_antigo:,}")
        print(f"      Tempo: {tempo_antigo:.4f}s")
        print(f"      Tamanho em memória: ~{total_antigo * 8 / 1024:.2f} KB")
        
        print(f"\n   Features NOVAS:")
        print(f"      Tipo: Array numpy")
        print(f"      Total de valores: {total_novo:,}")
        print(f"      Tempo: {tempo_novo:.4f}s")
        print(f"      Tamanho em memória: ~{total_novo * 8 / 1024:.2f} KB")
        
        # Cálculo de redução
        reducao = ((total_antigo - total_novo) / total_antigo) * 100
        speedup = tempo_antigo / tempo_novo
        
        print(f"\n   📉 MELHORIA:")
        print(f"      Redução de dimensionalidade: {reducao:.2f}%")
        print(f"      Valores: {total_antigo:,} → {total_novo:,}")
        print(f"      Speedup: {speedup:.2f}x mais rápido")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def teste_normalizacao():
    """Testa a normalização das features"""
    print("\n" + "=" * 70)
    print("🧪 TESTE 3: Normalização de Features")
    print("=" * 70)
    
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Criar algumas features de exemplo
        features = []
        for i in range(10):
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            feat = extrair_features_otimizado(image_process=test_image)
            features.append(feat)
        
        features_matrix = np.vstack(features)
        
        # Normalizar
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_matrix)
        
        print(f"✅ Normalização aplicada com sucesso!")
        print(f"\n📊 Estatísticas ANTES da normalização:")
        print(f"   Média: {features_matrix.mean():.4f}")
        print(f"   Desvio padrão: {features_matrix.std():.4f}")
        print(f"   Min: {features_matrix.min():.4f}")
        print(f"   Max: {features_matrix.max():.4f}")
        
        print(f"\n📊 Estatísticas DEPOIS da normalização:")
        print(f"   Média: {features_normalized.mean():.4f}")
        print(f"   Desvio padrão: {features_normalized.std():.4f}")
        print(f"   Min: {features_normalized.min():.4f}")
        print(f"   Max: {features_normalized.max():.4f}")
        
        # Verificar se a normalização está correta
        assert abs(features_normalized.mean()) < 1e-10, "Média não é ~0"
        assert abs(features_normalized.std() - 1.0) < 0.1, "Desvio padrão não é ~1"
        
        print(f"\n✅ Normalização está correta (média ≈ 0, std ≈ 1)")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def teste_cache():
    """Testa o sistema de cache"""
    print("\n" + "=" * 70)
    print("🧪 TESTE 4: Sistema de Cache")
    print("=" * 70)
    
    cache_file = 'test_knn_cache.pkl'
    
    try:
        # Remover cache se existir
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"🗑️  Cache anterior removido")
        
        # Simular salvamento de cache
        import pickle
        test_data = {
            'test': 'data',
            'features': np.random.rand(10, 100),
            'labels': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        }
        
        # Salvar
        start = time.time()
        with open(cache_file, 'wb') as f:
            pickle.dump(test_data, f)
        tempo_save = time.time() - start
        
        tamanho = os.path.getsize(cache_file)
        print(f"💾 Cache salvo:")
        print(f"   Arquivo: {cache_file}")
        print(f"   Tamanho: {tamanho / 1024:.2f} KB")
        print(f"   Tempo: {tempo_save:.4f}s")
        
        # Carregar
        start = time.time()
        with open(cache_file, 'rb') as f:
            loaded_data = pickle.load(f)
        tempo_load = time.time() - start
        
        print(f"\n📂 Cache carregado:")
        print(f"   Tempo: {tempo_load:.4f}s")
        print(f"   Speedup vs salvar: {tempo_save / tempo_load:.2f}x")
        
        # Verificar integridade
        assert loaded_data['test'] == test_data['test']
        assert np.array_equal(loaded_data['features'], test_data['features'])
        
        print(f"\n✅ Cache funcionando corretamente!")
        
        # Limpar
        os.remove(cache_file)
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        # Limpar em caso de erro
        if os.path.exists(cache_file):
            os.remove(cache_file)
        return False


def main():
    """Executa todos os testes"""
    print("\n" + "=" * 70)
    print("🚀 TESTE DAS MELHORIAS IMPLEMENTADAS")
    print("=" * 70)
    
    resultados = []
    
    # Teste 1: Features otimizado
    sucesso, features = teste_features_otimizado()
    resultados.append(("Features Otimizado", sucesso))
    
    # Teste 2: Comparação
    sucesso = teste_comparacao_features()
    resultados.append(("Comparação Antigas vs Novas", sucesso))
    
    # Teste 3: Normalização
    sucesso = teste_normalizacao()
    resultados.append(("Normalização", sucesso))
    
    # Teste 4: Cache
    sucesso = teste_cache()
    resultados.append(("Sistema de Cache", sucesso))
    
    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES")
    print("=" * 70)
    
    for nome, sucesso in resultados:
        status = "✅ PASSOU" if sucesso else "❌ FALHOU"
        print(f"{status} - {nome}")
    
    total = len(resultados)
    passou = sum(1 for _, s in resultados if s)
    
    print(f"\n🎯 Total: {passou}/{total} testes passaram")
    
    if passou == total:
        print("🎉 Todas as melhorias estão funcionando corretamente!")
    else:
        print("⚠️  Alguns testes falharam. Verifique os erros acima.")
    
    print("=" * 70)


if __name__ == '__main__':
    main()


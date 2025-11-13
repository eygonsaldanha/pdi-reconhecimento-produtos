"""
Script de validação do modelo KNN para reconhecimento de produtos.

Este script:
1. Carrega o dataset completo
2. Extrai features otimizadas
3. Valida o modelo com cross-validation
4. Testa diferentes valores de k
5. Gera relatórios e gráficos
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from db_common import select_data
from io_minio import get_single_object_img
from libs.knn_process import extrair_features_otimizado


def carregar_dataset():
    """Carrega dataset completo do banco de dados"""
    print("=" * 70)
    print("📥 CARREGANDO DATASET")
    print("=" * 70)
    
    df = select_data("""
        SELECT d.path_data, p.id_product, p.nm_product 
        FROM data d
        JOIN product_data pd ON pd.id_data = d.id_data
        JOIN product p ON p.id_product = pd.id_product
    """)
    
    print(f"✅ Dataset carregado: {len(df)} imagens")
    print(f"📊 Produtos únicos: {df['nm_product'].nunique()}")
    print(f"\nDistribuição por produto:")
    print(df['nm_product'].value_counts())
    
    return df


def extrair_features_dataset(df):
    """Extrai features de todo o dataset"""
    print("\n" + "=" * 70)
    print("🔍 EXTRAINDO FEATURES")
    print("=" * 70)
    
    X = []
    y = []
    produtos = []
    erros = 0
    
    for idx, row in df.iterrows():
        try:
            print(f"Processando {idx+1}/{len(df)}: {row['nm_product']}", end='\r')
            img = get_single_object_img(row['path_data'])
            features = extrair_features_otimizado(image_process=img)
            X.append(features)
            y.append(row['id_product'])
            produtos.append(row['nm_product'])
        except Exception as e:
            print(f"\n⚠️  Erro ao processar {row['path_data']}: {e}")
            erros += 1
    
    print(f"\n✅ Features extraídas: {len(X)} imagens")
    print(f"📐 Dimensão do vetor: {len(X[0])} valores")
    if erros > 0:
        print(f"⚠️  Erros: {erros} imagens")
    
    return np.array(X), np.array(y), produtos


def validar_knn(X, y, produtos, n_neighbors=5):
    """Valida o modelo KNN com train/test split"""
    print("\n" + "=" * 70)
    print(f"🤖 VALIDAÇÃO DO MODELO KNN (k={n_neighbors})")
    print("=" * 70)
    
    # Normalizar
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Verificar se há classes suficientes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("⚠️  Aviso: Apenas uma classe no dataset. Impossível validar.")
        return None, None, 0.0
    
    # Dividir treino/teste (stratify para manter proporção)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("⚠️  Não foi possível fazer split estratificado. Usando split simples.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42
        )
    
    # Treinar modelo
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X_train, y_train)
    
    # Avaliar
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    
    print(f"\n📊 Resultados:")
    print(f"   Acurácia no treino: {train_score:.4f} ({train_score*100:.2f}%)")
    print(f"   Acurácia no teste:  {test_score:.4f} ({test_score*100:.2f}%)")
    
    # Cross-validation (se houver dados suficientes)
    if len(X) >= 5:
        cv_folds = min(5, len(unique_classes))
        cv_scores = cross_val_score(knn, X_normalized, y, cv=cv_folds)
        print(f"\n🔄 Cross-validation ({cv_folds}-fold):")
        print(f"   Média: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        print(f"   Desvio padrão: {cv_scores.std():.4f}")
    
    # Predições
    y_pred = knn.predict(X_test)
    
    # Relatório de classificação
    print(f"\n📋 Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Matriz de confusão
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Matriz de Confusão - KNN (k={n_neighbors})', fontsize=14, fontweight='bold')
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print("💾 Matriz de confusão salva: confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Erro ao gerar matriz de confusão: {e}")
    
    return knn, scaler, test_score


def testar_diferentes_k(X, y, k_range=range(1, 21)):
    """Testa diferentes valores de k"""
    print("\n" + "=" * 70)
    print("📈 TESTANDO DIFERENTES VALORES DE K")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    unique_classes = np.unique(y)
    cv_folds = min(5, len(unique_classes), len(X) // 2)
    
    if cv_folds < 2:
        print("⚠️  Dataset muito pequeno para cross-validation")
        return 5
    
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        try:
            cv_scores = cross_val_score(knn, X_normalized, y, cv=cv_folds)
            score = cv_scores.mean()
            scores.append(score)
            print(f"k={k:2d}: acurácia={score:.4f} ({score*100:.2f}%)")
        except Exception as e:
            print(f"k={k:2d}: erro - {e}")
            scores.append(0.0)
    
    # Plot
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(k_range, scores, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Valor de k', fontsize=12)
        plt.ylabel('Acurácia (cross-validation)', fontsize=12)
        plt.title('Otimização do Hiperparâmetro k para KNN', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('k_optimization.png', dpi=150)
        print("\n💾 Gráfico salvo: k_optimization.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Erro ao gerar gráfico: {e}")
    
    best_k = list(k_range)[np.argmax(scores)]
    best_score = max(scores)
    print(f"\n✅ Melhor k: {best_k} (acurácia: {best_score:.4f} / {best_score*100:.2f}%)")
    
    return best_k


def visualizar_distribuicao_features(X, y, produtos):
    """Visualiza a distribuição das features"""
    print("\n" + "=" * 70)
    print("📊 ANÁLISE DE FEATURES")
    print("=" * 70)
    
    # Estatísticas básicas
    print(f"Shape: {X.shape}")
    print(f"Média: {X.mean():.4f}")
    print(f"Desvio padrão: {X.std():.4f}")
    print(f"Min: {X.min():.4f}")
    print(f"Max: {X.max():.4f}")
    
    # Histograma das primeiras features
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        features_to_plot = [0, 1, 2, 3]  # Primeiras 4 features (geométricas)
        feature_names = ['Área', 'Perímetro', 'Circularidade', 'Aspect Ratio']
        
        for idx, (ax, feat_idx, name) in enumerate(zip(axes.flatten(), features_to_plot, feature_names)):
            ax.hist(X[:, feat_idx], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Valor', fontsize=10)
            ax.set_ylabel('Frequência', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('features_distribution.png', dpi=150)
        print("💾 Distribuição de features salva: features_distribution.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Erro ao gerar visualização: {e}")


def main():
    """Função principal"""
    print("\n" + "=" * 70)
    print("🚀 VALIDAÇÃO DO SISTEMA KNN DE RECONHECIMENTO DE PRODUTOS")
    print("=" * 70)
    
    # Carregar dataset
    df = carregar_dataset()
    
    # Extrair features
    X, y, produtos = extrair_features_dataset(df)
    
    # Visualizar distribuição
    visualizar_distribuicao_features(X, y, produtos)
    
    # Testar diferentes valores de k
    best_k = testar_diferentes_k(X, y, k_range=range(1, min(21, len(X))))
    
    # Validar com melhor k
    knn, scaler, accuracy = validar_knn(X, y, produtos, n_neighbors=best_k)
    
    # Resumo final
    print("\n" + "=" * 70)
    print("✅ VALIDAÇÃO COMPLETA")
    print("=" * 70)
    print(f"📊 Dataset: {len(X)} imagens")
    print(f"📐 Features: {X.shape[1]} valores por imagem")
    print(f"🎯 Melhor k: {best_k}")
    print(f"🏆 Acurácia final: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n📁 Arquivos gerados:")
    print("   - confusion_matrix.png")
    print("   - k_optimization.png")
    print("   - features_distribution.png")
    print("=" * 70)


if __name__ == '__main__':
    main()


"""
Módulo de avaliação do modelo.
Gera matriz de confusão, acurácia por classe e análise de erros.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from typing import Tuple, List

from preprocessing import load_dataset
from feature_extraction import extract_features_from_dataset
from train import FruitClassifier


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plota e salva a matriz de confusão.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        classes: Lista de nomes das classes
        output_path: Caminho para salvar a figura (opcional)
        figsize: Tamanho da figura
    """
    # Calcula matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Cria figura
    plt.figure(figsize=figsize)

    # Plota heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Número de amostras"},
    )

    plt.title("Matriz de Confusão", fontsize=16, fontweight="bold")
    plt.ylabel("Classe Verdadeira", fontsize=12)
    plt.xlabel("Classe Predita", fontsize=12)
    plt.tight_layout()

    # Salva se path fornecido
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Matriz de confusão salva: {output_path}")

    plt.show()
    plt.close()

    return cm


def calculate_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]
) -> pd.DataFrame:
    """
    Calcula métricas detalhadas por classe.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        classes: Lista de nomes das classes

    Returns:
        DataFrame com métricas por classe
    """
    # Calcula precision, recall, f1-score por classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )

    # Calcula acurácia por classe
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Cria DataFrame
    metrics_df = pd.DataFrame(
        {
            "Classe": classes,
            "Acurácia": class_accuracy,
            "Precisão": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Amostras": support,
        }
    )

    return metrics_df


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    image_paths: List[str] = None,
) -> pd.DataFrame:
    """
    Analisa os erros de classificação.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        classes: Lista de nomes das classes
        image_paths: Lista de caminhos das imagens (opcional)

    Returns:
        DataFrame com análise dos erros
    """
    # Identifica erros
    errors_mask = y_true != y_pred
    error_indices = np.where(errors_mask)[0]

    if len(error_indices) == 0:
        print("Nenhum erro encontrado!")
        return pd.DataFrame()

    # Cria DataFrame com erros
    errors_data = []
    for idx in error_indices:
        error_info = {
            "Índice": idx,
            "Classe_Real": classes[y_true[idx]],
            "Classe_Predita": classes[y_pred[idx]],
        }
        if image_paths:
            error_info["Imagem"] = image_paths[idx]
        errors_data.append(error_info)

    errors_df = pd.DataFrame(errors_data)

    # Conta erros por par de classes
    error_pairs = errors_df.groupby(["Classe_Real", "Classe_Predita"]).size()
    error_pairs = error_pairs.sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("PARES DE CLASSES MAIS CONFUNDIDOS")
    print("=" * 60)
    print(error_pairs.head(10))

    return errors_df


def evaluate_model(
    model_path: str = "../models/knn_model.pkl",
    test_dataset_path: str = "../dataset-test",
    output_dir: str = "../results",
):
    """
    Avalia o modelo no conjunto de teste.

    Args:
        model_path: Caminho do modelo salvo
        test_dataset_path: Caminho do dataset de teste
        output_dir: Diretório para salvar resultados
    """
    print("\n" + "=" * 60)
    print("AVALIAÇÃO DO MODELO")
    print("=" * 60)

    # Carrega modelo
    print("\nCarregando modelo...")
    classifier = FruitClassifier.load_model(model_path)
    print(f"Modelo carregado: {model_path}")
    print(f"Classes: {classifier.classes}")

    # Carrega dataset de teste
    print("\n" + "=" * 60)
    print("CARREGANDO DATASET DE TESTE")
    print("=" * 60)
    images, labels, _ = load_dataset(test_dataset_path, size=classifier.image_size)

    # Extrai características
    print("\n" + "=" * 60)
    print("EXTRAINDO CARACTERÍSTICAS")
    print("=" * 60)
    X_test = extract_features_from_dataset(images)
    y_test = classifier.label_encoder.transform(labels)

    # Predições
    print("\n" + "=" * 60)
    print("REALIZANDO PREDIÇÕES")
    print("=" * 60)
    y_pred = classifier.knn.predict(X_test)

    # Acurácia geral
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia Geral: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Matriz de confusão
    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSÃO")
    print("=" * 60)
    cm_path = f"{output_dir}/confusion_matrix.png"
    cm = plot_confusion_matrix(y_test, y_pred, classifier.classes, output_path=cm_path)

    # Métricas por classe
    print("\n" + "=" * 60)
    print("MÉTRICAS POR CLASSE")
    print("=" * 60)
    metrics_df = calculate_class_metrics(y_test, y_pred, classifier.classes)
    print("\n", metrics_df.to_string(index=False))

    # Salva métricas
    metrics_path = f"{output_dir}/class_metrics.csv"
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMétricas salvas: {metrics_path}")

    # Relatório de classificação
    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 60)
    report = classification_report(y_test, y_pred, target_names=classifier.classes)
    print("\n", report)

    # Salva relatório
    report_path = f"{output_dir}/classification_report.txt"
    with open(report_path, "w") as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Acurácia Geral: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"Relatório salvo: {report_path}")

    # Análise de erros
    print("\n" + "=" * 60)
    print("ANÁLISE DE ERROS")
    print("=" * 60)
    errors_df = analyze_errors(y_test, y_pred, classifier.classes)

    if len(errors_df) > 0:
        errors_path = f"{output_dir}/errors_analysis.csv"
        errors_df.to_csv(errors_path, index=False)
        print(f"\nAnálise de erros salva: {errors_path}")
        print(
            f"Total de erros: {len(errors_df)}/{len(y_test)} ({len(errors_df)/len(y_test)*100:.2f}%)"
        )

    print("\n" + "=" * 60)
    print("AVALIAÇÃO CONCLUÍDA!")
    print("=" * 60)
    print(f"\nResultados salvos em: {output_dir}/")


if __name__ == "__main__":
    evaluate_model()

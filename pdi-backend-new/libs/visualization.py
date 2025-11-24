import matplotlib.pyplot as plt
import cv2
import numpy as np

def plotar_resultados_segmentacao(resultados):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(resultados['imagem_original'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(resultados['cinza'], cmap="gray")
    axes[0, 1].set_title("Escala de Cinza", fontsize=14, fontweight='bold')
    axes[0, 1].axis("off")

    axes[0, 2].imshow(resultados['gaussiano'], cmap="gray")
    axes[0, 2].set_title("Filtro Gaussiano", fontsize=14, fontweight='bold')
    axes[0, 2].axis("off")

    axes[1, 0].imshow(resultados['mascara_limpa'], cmap="gray")
    axes[1, 0].set_title("Máscara Limpa (Flood Fill)", fontsize=14, fontweight='bold')
    axes[1, 0].axis("off")

    axes[1, 1].imshow(resultados['bordas'], cmap="gray")
    axes[1, 1].set_title("Bordas Canny", fontsize=14, fontweight='bold')
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(resultados['imagem_contornos'], cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Contornos Filtrados", fontsize=14, fontweight='bold')
    axes[1, 2].axis("off")

    plt.suptitle("Pipeline Melhorado com Flood Fill", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def plotar_caracteristicas(resultados):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(resultados['imagem_contornos'], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Contornos Filtrados", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(resultados['lbp_imagem'], cmap="gray")
    axes[1].set_title("LBP (Local Binary Pattern)", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(resultados['hog_imagem'], cmap="inferno")
    axes[2].set_title("HOG (Histogram of Oriented Gradients)", fontsize=14, fontweight='bold')
    axes[2].axis("off")

    plt.suptitle("Características Extraídas - Pipeline Melhorado", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def plotar_histograma_lbp(resultados):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(resultados['lbp_histograma'])), resultados['lbp_histograma'])
    plt.title("Histograma LBP (Local Binary Pattern) - Pipeline Melhorado", fontsize=14, fontweight='bold')
    plt.xlabel("Bins do Histograma")
    plt.ylabel("Frequência Normalizada")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

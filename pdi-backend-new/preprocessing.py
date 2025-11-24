"""
Módulo de pré-processamento de imagens.
Responsável por carregar, redimensionar e normalizar imagens do dataset.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    return img


def resize_image(image: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def preprocess_image(image_path: str, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    img = load_image(image_path)
    img = resize_image(img, size)
    return img


def load_dataset(
    dataset_path: str, size: Tuple[int, int] = (128, 128)
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    dataset_dir = Path(dataset_path)
    images = []
    labels = []

    # Extensões de arquivo suportadas
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # Percorre cada pasta de classe
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"Carregando classe: {class_name}")

        # Carrega todas as imagens da classe
        image_count = 0
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                try:
                    img = preprocess_image(str(img_path), size)
                    images.append(img)
                    labels.append(class_name)
                    image_count += 1
                except Exception as e:
                    print(f"Erro ao carregar {img_path}: {e}")

        print(f"  -> {image_count} imagens carregadas")

    # Obtém lista única de classes
    unique_classes = sorted(list(set(labels)))

    print(f"\nTotal: {len(images)} imagens de {len(unique_classes)} classes")
    print(f"Classes: {unique_classes}")

    return images, labels, unique_classes

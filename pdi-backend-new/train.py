import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Tuple, Dict, Any
import time
from datetime import timedelta

from preprocessing import load_dataset
from feature_extraction import extract_features_from_dataset


class FruitClassifier:
    def __init__(self, image_size: Tuple[int, int] = (128, 128)):
        self.image_size = image_size
        self.knn = None
        self.label_encoder = LabelEncoder()
        self.classes = None
        self.best_params = None

    def prepare_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Carrega imagens
        images, labels, self.classes = load_dataset(dataset_path, size=self.image_size)

        # Codifica labels
        y = self.label_encoder.fit_transform(labels)

        # Extrai características
        X = extract_features_from_dataset(images)

        return X, y

    def tune_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("AJUSTE DE HIPERPARÂMETROS")
        print("=" * 60)

        # Define grid de parâmetros para testar
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        }

        print(
            f"Testando {np.prod([len(v) for v in param_grid.values()])} combinações..."
        )
        print(f"Parâmetros: {param_grid}")

        # Cria modelo base
        knn_base = KNeighborsClassifier()

        # Grid search
        start_time = time.time()
        grid_search = GridSearchCV(
            knn_base, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
        )

        grid_search.fit(X, y)
        elapsed_time = time.time() - start_time

        print(f"\nTempo de busca: {elapsed_time:.2f}s")
        print(f"\nMelhores parâmetros: {grid_search.best_params_}")
        print(f"Melhor acurácia (CV): {grid_search.best_score_:.4f}")

        self.best_params = grid_search.best_params_
        return grid_search.best_params_

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] = None):
        if params is None:
            params = {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}

        print(f"\nParâmetros do modelo: {params}")

        # Cria e treina modelo
        self.knn = KNeighborsClassifier(**params)

        start_time = time.time()
        self.knn.fit(X, y)
        elapsed_time = time.time() - start_time

        print(f"Tempo de treinamento: {elapsed_time:.2f}s")

        # Cross-validation para estimar performance
        cv_scores = cross_val_score(self.knn, X, y, cv=5, scoring="accuracy")
        print(f"Cross-validation (5-fold):")
        print(f"  - Acurácia média: {cv_scores.mean():.4f}")
        print(f"  - Desvio padrão: {cv_scores.std():.4f}")
        print(f"  - Scores: {cv_scores}")

    def save_model(self, output_path: str = "../models/knn_model.pkl"):
        if self.knn is None:
            raise ValueError("Modelo não foi treinado ainda!")

        model_data = {
            "knn": self.knn,
            "label_encoder": self.label_encoder,
            "classes": self.classes,
            "best_params": self.best_params,
            "image_size": self.image_size,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, output_path)

        print(f"\n{'=' * 60}")
        print(f"MODELO SALVO: {output_path}")
        print(f"{'=' * 60}")

    def load_model(model_path: str = "../models/knn_model.pkl") -> "FruitClassifier":
        model_data = joblib.load(model_path)

        classifier = FruitClassifier(image_size=model_data["image_size"])
        classifier.knn = model_data["knn"]
        classifier.label_encoder = model_data["label_encoder"]
        classifier.classes = model_data["classes"]
        classifier.best_params = model_data["best_params"]

        return classifier


def main():
    start_time = time.time()

    print("\n" + "=" * 60)
    print("SISTEMA DE CLASSIFICAÇÃO DE FRUTAS - TREINAMENTO")
    print("=" * 60)

    # Inicializa classificador
    classifier = FruitClassifier(image_size=(128, 128))

    # Prepara dados de treino
    X_train, y_train = classifier.prepare_data("../dataset")

    # Ajusta hiperparâmetros
    best_params = classifier.tune_hyperparameters(X_train, y_train, cv=5)

    # Treina modelo final
    classifier.train(X_train, y_train, params=best_params)

    # Salva modelo
    classifier.save_model("../models/knn_model.pkl")

    elapsed_seconds = int(time.time() - start_time)
    elapsed_time = str(timedelta(seconds=elapsed_seconds))

    print("\n" + "=" * 60)
    print(f"TREINAMENTO CONCLUÍDO! {elapsed_time}")
    print("=" * 60)


if __name__ == "__main__":
    main()

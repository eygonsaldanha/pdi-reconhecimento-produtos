from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Any, Dict
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import cv2
from pathlib import Path
import io
from typing import Dict, Any
import uvicorn
import joblib

from train import FruitClassifier
from feature_extraction import extract_features

MODEL_PATH = Path(__file__).parent.parent / "models" / "knn_model.pkl"


def find_clazz_by_idx(idx):
    for clazz_and_price in clazz_and_prices:
        if clazz_and_price["idx"] == idx:
            return clazz_and_price


def find_clazz_by_name(name):
    for clazz_and_price in clazz_and_prices:
        if clazz_and_price["name"].upper() == name.upper():
            return clazz_and_price


app = FastAPI(
    title="Classificador de Frutas API",
    description="API para classificação de frutas usando PDI e KNN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: FruitClassifier = None
clazz_and_prices = None


@app.on_event("startup")
async def load_model():
    global classifier
    global clazz_and_prices

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")

        classifier, clazz_and_prices = FruitClassifier.load_model(str(MODEL_PATH))
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        raise


def process_uploaded_image(file_bytes: bytes, image_size: tuple) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Não foi possível decodificar a imagem")
    return cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...), not_is_fruit: List[int] = Form([])
) -> Dict[str, Any]:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo inválido. Esperado imagem, recebido: {file.content_type}",
        )

    not_is_fruit = [find_clazz_by_idx(i)["name"] for i in not_is_fruit]

    try:
        contents = await file.read()

        img = process_uploaded_image(contents, classifier.image_size)

        features = extract_features(img)
        features = features.reshape(1, -1)

        probas = classifier.knn.predict_proba(features)[0]
        classes = classifier.classes

        for i, c in enumerate(classes):
            if c in not_is_fruit:
                probas[i] = -1.0

        proba_sum = probas.sum()

        if proba_sum > 0:
            probas = probas / proba_sum
            prediction = probas.argmax()

        else:
            valid_indices = [i for i, c in enumerate(classes) if c not in not_is_fruit]

            if len(valid_indices) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Nenhuma classe válida disponível para escolher.",
                )

            prediction = valid_indices[0]

        predicted_label = classifier.label_encoder.inverse_transform([prediction])[0]
        predicted_fruit = find_clazz_by_name(predicted_label)
        probabilities = {classes[i]: float(probas[i]) for i in range(len(classes))}

        return JSONResponse(
            content={
                "idx_fruit": predicted_fruit["idx"],
                "predicted_fruit": predicted_fruit["name"],
                "price": predicted_fruit["price"],
                "confidence": float(probas[prediction]) if proba_sum > 0 else 0.0,
                "probabilities": probabilities,
            }
        )

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar imagem: {str(e)}"
        )


def start_api(host: str = "0.0.0.0", port: int = 8000):
    print("\n" + "=" * 60)
    print("INICIANDO API DE CLASSIFICAÇÃO DE FRUTAS")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api()

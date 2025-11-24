from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from pathlib import Path
import io
from typing import Dict, Any
import uvicorn
import joblib
import random

from train import FruitClassifier
from feature_extraction import extract_features

MODEL_PATH = Path(__file__).parent.parent / "models" / "knn_model.pkl"


def find_clazz_by_name(name):
    model_path = "../models/knn_model.pkl"
    model_data = joblib.load(model_path)

    clazz_and_prices = [
        {"idx": idx, "name": clazz, "price": round(5 + (random.random() * 15), 2)}
        for idx, clazz in enumerate(model_data["classes"])
    ]
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


@app.on_event("startup")
async def load_model():
    global classifier

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")

        classifier = FruitClassifier.load_model(str(MODEL_PATH))
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
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo inválido. Esperado imagem, recebido: {file.content_type}",
        )

    try:
        contents = await file.read()

        img = process_uploaded_image(contents, classifier.image_size)

        features = extract_features(img)
        features = features.reshape(1, -1)

        prediction = classifier.knn.predict(features)[0]
        probas = classifier.knn.predict_proba(features)[0]

        predicted_fruit = classifier.label_encoder.inverse_transform([prediction])[0]
        predicted_fruit = find_clazz_by_name(predicted_fruit)

        return JSONResponse(
            content={
                "predicted_fruit": predicted_fruit["name"],
                "price": predicted_fruit["price"],
                "confidence": float(probas[prediction]),
                "probabilities": {
                    classifier.classes[i]: float(probas[i])
                    for i in range(len(classifier.classes))
                },
            }
        )

    except Exception as e:
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

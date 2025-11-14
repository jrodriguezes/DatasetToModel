# ============================================================
# API PARA EXPONER EL MODELO DE HIGH_VALUE VIA HTTP
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd

# ------------------------------------------------------------
# 1. CARGA DEL MODELO Y DEL SCHEMA
# ------------------------------------------------------------

# Cargar esquema de columnas desde schema/schema.json
with open("schema/schema.json", "r") as f:
    schema = json.load(f)

numeric_features = schema["numeric"]
categorical_features = schema["categorical"]
TARGET = schema["target"]

# Cargar el modelo entrenado (Pipeline con preprocesador + RandomForest)
model = joblib.load("models/model.joblib")

# ------------------------------------------------------------
# 2. DEFINICION DE LA ESTRUCTURA DE ENTRADA (REQUEST BODY)
# ------------------------------------------------------------


class RetailInput(BaseModel):
    age: int
    quantity: int
    price_unit: float
    total_amount: float
    gender: str
    product_cat: str


# ------------------------------------------------------------
# 3. CREACION DE LA APLICACION FASTAPI
# ------------------------------------------------------------

app = FastAPI(
    title="API de Clasificación de Compras High Value",
    description="API que usa un modelo de Machine Learning para predecir si una compra es de alto valor (high_value).",
    version="1.0.0",
)

# ------------------------------------------------------------
# 4. ENDPOINT DE PRUEBA (HEALTHCHECK)
# ------------------------------------------------------------


@app.get("/health")
def healthcheck():
    """
    Endpoint simple para verificar que la API está arriba.
    """
    return {"status": "ok"}


# ------------------------------------------------------------
# 5. ENDPOINT DE PREDICCION
# ------------------------------------------------------------


@app.post("/predict")
def predict(input_data: RetailInput):
    """
    Recibe los datos de una compra y devuelve:
    - high_value: 0 o 1
    - probability: probabilidad de que sea high_value = 1
    """

    # Convertir el input (Pydantic) a dict y luego a DataFrame
    data_dict = input_data.dict()
    df_input = pd.DataFrame([data_dict])

    # Asegurar orden de columnas igual al usado en el entrenamiento
    all_features = numeric_features + categorical_features
    df_input = df_input[all_features]

    # Obtener predicción (0 o 1)
    y_pred = model.predict(df_input)[0]

    # Obtener probabilidad de la clase 1 (high_value = 1)
    y_prob = model.predict_proba(df_input)[0][1]

    return {"input": data_dict, "prediction": int(y_pred), "probability": float(y_prob)}

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import pandas as pd  # Manejo de datos en tablas (leer CSV, filtrar, describir)

import numpy as np  # Calculos numericos y manejo de arreglos

import matplotlib.pyplot as plt  # Graficos simples (lineas, barras, histogramas)

import seaborn as sns  # Graficos estadisticos mas atractivos (heatmaps, distribuciones)

# ============================================================
# LIBRERIAS PRINCIPALES DE MACHINE LEARNING (Scikit-Learn)
# ============================================================

from sklearn.model_selection import train_test_split  # Division del dataset en entrenamiento y prueba

from sklearn.preprocessing import StandardScaler, OneHotEncoder
# StandardScaler  -> normaliza valores numericos
# OneHotEncoder -> convierte categorias en vectores numericos

from sklearn.compose import ColumnTransformer  # Aplica distintos preprocesamientos según el tipo de columna

from sklearn.pipeline import Pipeline  # Encadena el preprocesamiento + el modelo en un solo flujo

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
# Metricas de evaluacion: matriz de confusion, precisión/recall/F1, curva ROC y area bajo la curva

from sklearn.ensemble import RandomForestClassifier  # Modelo de clasificacion basado en muchos arboles de decision

# ============================================================
# LIBRERIAS PARA GUARDAR Y CARGAR MODELOS
# ============================================================

import joblib  # Guardar y cargar modelos entrenados en archivos .joblib

import json  # Manejar archivos JSON (configuracion, schema del modelo)


# ============================================================
# 2. CARGA DEL DATASET
# ============================================================

print("\nCargando dataset...")
df = pd.read_csv("data/retail_sales.csv")
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())


# ============================================================
# 3. CREACION DEL TARGET HIGH_VALUE
# ============================================================

print("\nCreando columna objetivo 'high_value'...")
df["high_value"] = (df["total_amount"] >= 300).astype(int)
print(df[["total_amount", "high_value"]].head())


# ============================================================
# 4. EXPLORACION DE DATOS
# ============================================================

print("\nDescripción estadística:")
print(df.describe())

print("\nValores nulos por columna:")
print(df.isnull().sum())

# Mapa de correlacion
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()


# ============================================================
# 5. DEFINICION DE VARIABLES
# ============================================================

# En esta sección se definen las variables de entrada (X) y la variable objetivo (y)
# que el modelo utilizará para aprender a realizar predicciones.

TARGET = "high_value"

numeric_features = ["age", "quantity", "price_unit", "total_amount"]
categorical_features = ["gender", "product_cat"]

print("\nColumnas numéricas:", numeric_features)
print("Columnas categóricas:", categorical_features)
print("Columna objetivo:", TARGET)

X = df[numeric_features + categorical_features]
y = df[TARGET]


# ============================================================
# 6. SPLIT TRAIN/TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTamaños:")
print("Train:", X_train.shape)
print("Test: ", X_test.shape)


# ============================================================
# 7. PIPELINE (PREPROCESAMIENTO + MODELO)
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nEntrenando modelo...")
pipeline.fit(X_train, y_train)
print("Entrenamiento finalizado.")


# ============================================================
# 8. EVALUACION
# ============================================================

print("\n===== EVALUACIÓN =====")
y_pred = pipeline.predict(X_test)

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Curva ROC
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.show()


# ============================================================
# 9. EXPORTACION DEL MODELO Y ESQUEMA
# ============================================================

print("\nGuardando modelo en 'models/model.joblib'...")
joblib.dump(pipeline, "models/model.joblib")

schema = {
    "numeric": numeric_features,
    "categorical": categorical_features,
    "target": TARGET
}

with open("schema/schema.json", "w") as f:
    json.dump(schema, f, indent=4)

print("Schema guardado en 'schema/schema.json'")

print("\n==== PROCESO FINALIZADO ====")
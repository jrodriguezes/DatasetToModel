# Proyecto de Modelado: Clasificación de Compras de Alto Valor

Este proyecto entrena un modelo de Machine Learning para clasificar compras en dos clases:

- `high_value = 1` → compra de alto valor (monto total ≥ 300)
- `high_value = 0` → compra de valor normal

El flujo completo incluye:
1. Carga del dataset.
2. Creación de la columna objetivo `high_value`.
3. Exploración de datos (estadísticas y matriz de correlación).
4. Definición de variables de entrada (features) y variable objetivo (target).
5. División en conjuntos de entrenamiento y prueba (train/test split).
6. Construcción de un **Pipeline** con preprocesamiento + modelo.
7. Entrenamiento del modelo `RandomForestClassifier`.
8. Evaluación del modelo (matriz de confusión, reporte de clasificación, curva ROC).
9. Exportación del modelo entrenado y el esquema de columnas.

---

## Estructura del proyecto

```txt
.
├── data/
│   └── retail_sales.csv        # Dataset de ventas minoristas
├── models/
│   └── model.joblib            # Modelo entrenado (se genera al ejecutar el script)
├── schema/
│   └── schema.json             # Definición de columnas numéricas/categóricas/target
├── train_model.py              # Script principal de entrenamiento
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo

Dependencias:
pip install pandas numpy matplotlib seaborn scikit-learn joblib
pip install fastapi uvicorn

Levantar la API
uvicorn api:app --reload
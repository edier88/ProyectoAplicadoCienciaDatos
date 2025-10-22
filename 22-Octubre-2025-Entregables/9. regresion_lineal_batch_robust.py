#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regresion_lineal_batch_robust_v4.py
------------------------------------
Versión final adaptada para datasets WiFi Cali con:
- TARGET = "target"
- Columnas baseline tipo "USAGE.KB_t-1"
- Escalado de features con RobustScaler
- Escalado interno del target con TransformedTargetRegressor
- Sin limpieza de NaN (ya se maneja externamente)

Autor: Equipo Tesis WiFi Cali
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

# === CONFIGURACIÓN GLOBAL ===
TRAIN_DIR = "train_windowed_kb"
TEST_DIR = "test_windowed_kb"
TARGET = "target"
OUTPUT_CSV = "resultados_regresion_todas_zonas-190CT2025.csv"
MODELOS_DIR = "modelos_regresion-190CT2025"

os.makedirs(MODELOS_DIR, exist_ok=True)

# === FUNCIÓN DE MÉTRICAS ===
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# === PROCESAMIENTO DE CADA ZONA ===
def procesar_zona(nombre_zona, train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Separar X e y
    X_train = df_train.drop(columns=[TARGET])
    y_train = df_train[TARGET]
    X_test = df_test.drop(columns=[TARGET])
    y_test = df_test[TARGET]

    # Pipeline con escalado de features y target
    model = Pipeline([
        ("scaler", RobustScaler(quantile_range=(25.0, 75.0))),
        ("regressor", TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=RobustScaler(quantile_range=(25.0, 75.0)))
        )
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas del modelo
    mae, rmse, r2 = compute_metrics(y_test, y_pred)

    # === BASELINE NAIVE ===
    baseline_col = None
    for col in X_test.columns:
        name = col.strip().lower()
        if "usage" in name and ("t-1" in name or "t_1" in name):
            baseline_col = col
            break

    if baseline_col:
        y_pred_baseline = X_test[baseline_col].values
        mae_b, rmse_b, r2_b = compute_metrics(y_test, y_pred_baseline)
    else:
        mae_b = rmse_b = r2_b = None

    # Guardar modelo
    modelo_path = os.path.join(MODELOS_DIR, f"{nombre_zona}.joblib")
    dump(model, modelo_path)

    print(f"✅ {nombre_zona} procesada | MAE: {mae:.4f} | R2: {r2:.4f}")

    return {
        "Zona": nombre_zona,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE_baseline": mae_b,
        "RMSE_baseline": rmse_b,
        "R2_baseline": r2_b
    }

# === FUNCIÓN PRINCIPAL ===
def main():
    resultados = []
    archivos_train = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".csv")]

    for archivo in archivos_train:
        train_path = os.path.join(TRAIN_DIR, archivo)
        test_path = os.path.join(TEST_DIR, archivo)
        nombre_zona = archivo.replace(".csv", "")

        if not os.path.exists(test_path):
            print(f"⚠️ No se encontró archivo test para {archivo}, se omite.")
            continue

        try:
            resultado = procesar_zona(nombre_zona, train_path, test_path)
            resultados.append(resultado)
        except Exception as e:
            print(f"❌ Error procesando {archivo}: {e}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(OUTPUT_CSV, index=False)
        print(f"\n📊 Resultados guardados en: {OUTPUT_CSV}")
    else:
        print("⚠️ No se generaron resultados.")

if __name__ == "__main__":
    main()

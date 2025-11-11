# -*- coding: utf-8 -*-
"""
Script: svr_por_zona_desescalado.py
----------------------------------------
Objetivo:
    Entrenar un modelo de Regresión de Vectores de Soporte (SVR)
    individual por cada zona WiFi, utilizando los archivos ventaneados
    escalados (train/test). El script aplica desescalado con el scaler
    correspondiente, calcula métricas de desempeño (MAE, RMSE, R²)
    en unidades reales (KB), y consolida los resultados globales.

Entradas:
    - train_windowed/ : contiene los datasets de entrenamiento por zona.
    - test_windowed/  : contiene los datasets de prueba por zona.
    - scalers_original/ : contiene los .pkl de los escaladores RobustScaler.

Salidas:
    - resultados_svr_por_zona/ : CSV de predicciones y métricas por zona.
    - resumen_metricas_svr.csv : consolidado global de métricas.

Autores:
    Equipo de Tesis - Maestría en Ciencia de Datos (Pamartin & Edier)
Fecha:
    2025-11-10

Cumple con:
    - Estándar PEP 8.
    - Recomendaciones de la reunión académica del 22 de octubre de 2025.
"""

# ======================================================
# 🔹 1. Importación de librerías necesarias
# ======================================================

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ======================================================
# 🔹 2. Configuración de rutas de carpetas
# ======================================================

TRAIN_DIR = Path("train_windowed")
TEST_DIR = Path("test_windowed")
SCALERS_DIR = Path("scalers_original")
OUTPUT_DIR = Path("resultados_svr_por_zona")

# Crear la carpeta de salida si no existe
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================================================
# 🔹 3. Función para calcular métricas de desempeño
# ======================================================

def calcular_metricas(y_true, y_pred):
    """
    Calcula las métricas MAE, RMSE y R² entre los valores reales y predichos.

    Parámetros
    ----------
    y_true : np.ndarray
        Valores reales desescalados (en KB).
    y_pred : np.ndarray
        Valores predichos desescalados (en KB).

    Retorna
    -------
    dict
        Diccionario con las métricas calculadas.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ======================================================
# 🔹 4. Inicialización de lista de resultados globales
# ======================================================

resultados_globales = []

# ======================================================
# 🔹 5. Bucle principal: recorrer zonas
# ======================================================

for archivo_train in sorted(TRAIN_DIR.glob("*.csv")):

    # Derivar nombre base de la zona
    zona_base = archivo_train.stem
    archivo_test = TEST_DIR / f"{zona_base}.csv"
    scaler_path = SCALERS_DIR / f"{zona_base}.pkl"

    print(f"\n📡 Procesando zona: {zona_base}")

    # Verificar existencia de archivos requeridos
    if not archivo_test.exists():
        print(f"⚠️ No se encontró el archivo de test para {zona_base}.")
        continue
    if not scaler_path.exists():
        print(f"⚠️ No se encontró el scaler para {zona_base}.")
        continue

    # ------------------------------------------------------
    # 5.1 Cargar datasets
    # ------------------------------------------------------
    df_train = pd.read_csv(archivo_train)
    df_test = pd.read_csv(archivo_test)

    # Separar variables predictoras y target
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # ------------------------------------------------------
    # 5.2 Cargar el scaler correspondiente a la zona
    # ------------------------------------------------------
    scaler_y = joblib.load(scaler_path)

    # ------------------------------------------------------
    # 5.3 Entrenar el modelo SVR (individual por zona)
    # ------------------------------------------------------
    model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    model.fit(X_train, y_train)

    # ------------------------------------------------------
    # 5.4 Predicción sobre el conjunto de prueba
    # ------------------------------------------------------
    y_pred_scaled = model.predict(X_test)

    # ------------------------------------------------------
    # 5.5 Desescalado de valores reales y predichos
    # ------------------------------------------------------
    y_pred_descaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_descaled = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).ravel()

    # ------------------------------------------------------
    # 5.6 Cálculo de métricas en KB
    # ------------------------------------------------------
    metricas = calcular_metricas(y_test_descaled, y_pred_descaled)
    metricas["zona"] = zona_base

    resultados_globales.append(metricas)

    # ------------------------------------------------------
    # 5.7 Guardar predicciones por zona
    # ------------------------------------------------------
    df_resultado = pd.DataFrame({
        "y_real_KB": y_test_descaled,
        "y_predicho_KB": y_pred_descaled
    })
    df_resultado["error_absoluto_KB"] = abs(df_resultado["y_real_KB"] - df_resultado["y_predicho_KB"])
    df_resultado["error_relativo_%"] = (
        abs(df_resultado["y_real_KB"] - df_resultado["y_predicho_KB"]) /
        df_resultado["y_real_KB"].replace(0, np.nan)
    ) * 100

    # Guardar resultados individuales
    salida_zona = OUTPUT_DIR / f"predicciones_{zona_base}.csv"
    df_resultado.to_csv(salida_zona, index=False, encoding="utf-8")

    print(f"  ✅ Zona {zona_base} procesada correctamente.")

# ======================================================
# 🔹 6. Consolidado global: métricas promedio y desviación
# ======================================================

df_metricas = pd.DataFrame(resultados_globales)

if not df_metricas.empty:
    # Calcular media y desviación estándar de las métricas
    resumen = {
        "Modelo": "SVR (kernel='rbf')",
        "Media_MAE_KB": df_metricas["MAE"].mean(),
        "Desv_MAE_KB": df_metricas["MAE"].std(),
        "Media_RMSE_KB": df_metricas["RMSE"].mean(),
        "Desv_RMSE_KB": df_metricas["RMSE"].std(),
        "Media_R2": df_metricas["R2"].mean(),
        "Desv_R2": df_metricas["R2"].std(),
    }

    df_resumen = pd.DataFrame([resumen])

    # Guardar archivo global de métricas
    resumen_path = OUTPUT_DIR / "resumen_metricas_svr.csv"
    df_resumen.to_csv(resumen_path, index=False, encoding="utf-8")

    print("\n📊 Resumen global de métricas:")
    print(df_resumen)

else:
    print("⚠️ No se generaron métricas; verifique los archivos de entrada.")

# ======================================================
# 🔹 7. Finalización del proceso
# ======================================================
print("\n🏁 Proceso completado: todos los modelos SVR entrenados y evaluados.")
print(f"📂 Resultados guardados en: {OUTPUT_DIR}")

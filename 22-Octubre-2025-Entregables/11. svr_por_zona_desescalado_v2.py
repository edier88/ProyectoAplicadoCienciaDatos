# -*- coding: utf-8 -*-
"""
Script: 11.svr_por_zona_desescalado_v5.py
------------------------------------------
Objetivo:
    Entrenar un SVR (kernel RBF) por zona usando datasets ventaneados,
    desescalar correctamente la variable objetivo (USAGE.KB) con el
    escalador por zona de 3 variables (_3vars.pkl), calcular métricas en KB
    y generar un resumen global. Las zonas que no cumplan condiciones
    mínimas (sin _3vars, sin datos tras limpieza, etc.) se omiten y se
    registran en un log.

Entradas:
    - train_windowed/ : CSVs de entrenamiento por zona (ventaneados).
    - test_windowed/  : CSVs de prueba por zona (ventaneados).
    - scalers_original/ : escaladores por zona; se requiere *_3vars.pkl.

Salidas:
    - resultados_svr_por_zona/predicciones_<zona>.csv
    - resultados_svr_por_zona/resumen_metricas_svr.csv
    - logs/zonas_omitidas.csv

Autor:
    Equipo de Tesis - Maestría en Ciencia de Datos (Pamartin & Edier)
Fecha:
    2025-11-11
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================
# 🔧 Configuración de rutas
# ==========================
TRAIN_DIR = Path("train_windowed")
TEST_DIR = Path("test_windowed")
SCALERS_DIR = Path("scalers_original")
OUTPUT_DIR = Path("resultados_svr_por_zona")
LOG_DIR = Path("logs")

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ==========================
# 📏 Utilidades
# ==========================
def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE, RMSE y R² en KB."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ==========================
# 🚀 Proceso principal
# ==========================
def main() -> None:
    resultados_globales = []
    zonas_omitidas = []

    archivos_train = sorted(TRAIN_DIR.glob("*.csv"))

    if not archivos_train:
        print("⚠️  No se encontraron archivos en train_windowed/.")
        return

    print(f"📊 Zonas a procesar: {len(archivos_train)}")

    for archivo_train in archivos_train:
        zona_base = archivo_train.stem
        archivo_test = TEST_DIR / f"{zona_base}.csv"

        print(f"\n📡 Procesando zona: {zona_base}")

        # --------------------------
        # 1) Validaciones de pares
        # --------------------------
        if not archivo_test.exists():
            msg = "No existe CSV en test_windowed"
            print(f"  ⚠️  {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        # Se exige el escalador de 3 variables por consistencia del desescalado
        scaler_path_3 = SCALERS_DIR / f"{zona_base}_3vars.pkl"
        if not scaler_path_3.exists():
            msg = "Falta escalador _3vars.pkl (re-ejecutar 6.scale_with_robust_v3_hybrid.py)"
            print(f"  ⚠️  {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        # --------------------------
        # 2) Cargar datos
        # --------------------------
        try:
            df_train = pd.read_csv(archivo_train)
            df_test = pd.read_csv(archivo_test)
        except Exception as e:
            msg = f"Error al leer CSVs: {e}"
            print(f"  ❌ {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        if df_train.empty or df_test.empty:
            msg = "CSV vacío (train o test)"
            print(f"  ⚠️  {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        # Asumimos target en la última columna del ventaneo (y)
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

        # --------------------------
        # 3) Limpieza de NaN
        # --------------------------
        train_concat = pd.concat([X_train, y_train], axis=1)
        test_concat = pd.concat([X_test, y_test], axis=1)

        n_nan_train = int(train_concat.isna().any(axis=1).sum())
        n_nan_test = int(test_concat.isna().any(axis=1).sum())
        if n_nan_train or n_nan_test:
            print(f"  ⚠️  {zona_base}: NaN detectados → train:{n_nan_train}, test:{n_nan_test}")

        train_concat = train_concat.dropna().reset_index(drop=True)
        test_concat = test_concat.dropna().reset_index(drop=True)

        if train_concat.empty or test_concat.empty:
            msg = "Sin filas válidas tras limpieza de NaN"
            print(f"  ⚠️  {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        X_train = train_concat.iloc[:, :-1]
        y_train = train_concat.iloc[:, -1]
        X_test = test_concat.iloc[:, :-1]
        y_test = test_concat.iloc[:, -1]

        print(f"  ✓  {zona_base}: tamaños → train={len(X_train)}, test={len(X_test)}")

        # --------------------------
        # 4) Cargar escalador 3 vars
        # --------------------------
        try:
            scaler_3 = joblib.load(scaler_path_3)
            # Por construcción del v3_hybrid, el orden es:
            # ['NUMERO.CONEXIONES', 'USAGE.KB', 'PORCENTAJE.USO']
            USAGE_IDX = 1  # posición de USAGE.KB en el scaler _3vars
            print(f"  🎯 Usando scaler 3-variables: {scaler_path_3.name}")
        except Exception as e:
            msg = f"No se pudo cargar _3vars.pkl: {e}"
            print(f"  ❌ {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        # --------------------------
        # 5) Entrenar SVR por zona
        # --------------------------
        try:
            model = SVR(kernel="rbf", C=1000, gamma="scale", epsilon=0.01)
            model.fit(X_train, y_train)
        except Exception as e:
            msg = f"Fallo entrenando SVR: {e}"
            print(f"  ❌ {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        # --------------------------
        # 6) Predicción y desescalado
        # --------------------------
        try:
            y_pred_scaled = model.predict(X_test)

            n_pred = len(y_pred_scaled)
            n_true = len(y_test)

            # Matrices placeholder de 3 columnas para aplicar inverse_transform
            ph_pred = np.zeros((n_pred, 3), dtype=float)
            ph_true = np.zeros((n_true, 3), dtype=float)

            # Insertamos en la columna USAGE.KB (índice 1)
            ph_pred[:, USAGE_IDX] = y_pred_scaled
            ph_true[:, USAGE_IDX] = y_test.values

            des_pred_all = scaler_3.inverse_transform(ph_pred)
            des_true_all = scaler_3.inverse_transform(ph_true)

            y_pred_descaled = des_pred_all[:, USAGE_IDX]
            y_true_descaled = des_true_all[:, USAGE_IDX]
        except Exception as e:
            msg = f"Error en desescalado: {e}"
            print(f"  ❌ {zona_base}: {msg}. Se omite.")
            zonas_omitidas.append({"zona": zona_base, "motivo": msg})
            continue

        # --------------------------
        # 7) Métricas y guardado por zona
        # --------------------------
        metricas = calcular_metricas(y_true_descaled, y_pred_descaled)
        metricas["zona"] = zona_base
        resultados_globales.append(metricas)

        df_out = pd.DataFrame({
            "y_real_KB": y_true_descaled,
            "y_predicho_KB": y_pred_descaled
        })
        df_out["error_absoluto_KB"] = np.abs(df_out["y_real_KB"] - df_out["y_predicho_KB"])
        df_out["error_relativo_%"] = (
            df_out["error_absoluto_KB"] / df_out["y_real_KB"].replace(0, np.nan)
        ) * 100

        out_path = OUTPUT_DIR / f"predicciones_{zona_base}.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  ✅ Guardado: {out_path.name}")

    # --------------------------
    # 8) Resumen global + logs
    # --------------------------
    if resultados_globales:
        df_m = pd.DataFrame(resultados_globales)
        resumen = pd.DataFrame([{
            "Modelo": "SVR (RBF) - 3 variables",
            "Media_MAE_KB": df_m["MAE"].mean(),
            "Desv_MAE_KB": df_m["MAE"].std(),
            "Media_RMSE_KB": df_m["RMSE"].mean(),
            "Desv_RMSE_KB": df_m["RMSE"].std(),
            "Media_R2": df_m["R2"].mean(),
            "Desv_R2": df_m["R2"].std(),
            "Zonas_evaluadas": len(df_m)
        }])
        resumen_path = OUTPUT_DIR / "resumen_metricas_svr.csv"
        resumen.to_csv(resumen_path, index=False, encoding="utf-8")
        print("\n📊 Resumen global de métricas:\n", resumen)
    else:
        print("\n⚠️  No se generaron métricas (todas las zonas fueron omitidas).")

    if zonas_omitidas:
        df_skip = pd.DataFrame(zonas_omitidas)
        skip_path = LOG_DIR / "zonas_omitidas.csv"
        df_skip.to_csv(skip_path, index=False, encoding="utf-8")
        print(f"📝 Log de zonas omitidas → {skip_path}")


if __name__ == "__main__":
    main()

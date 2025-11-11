# ===========================================================
# train_random_forest_por_zona_v2.py
# Entrena un modelo Random Forest independiente para cada zona WiFi
# usando los datasets ventaneados (train_windowed/ y test_windowed/).
# Ahora incluye cálculo de R².
# 21-octubre-2025 PAM-EDIER 
# ===========================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== CONFIGURACIÓN =====
TRAIN_DIR = "train_windowed/"
TEST_DIR = "test_windowed/"
TARGET = "target"
N_ESTIMATORS = 300
RANDOM_STATE = 42
N_JOBS = -1  # usa todos los núcleos disponibles
RESULTADOS_PATH = "resultados_por_zona-con-R-21-10-2025.csv"
PRED_DIR = "predicciones_por_zona-con-R-21-10-2025"
GRAF_DIR = "graficas_por_zona-con-R-21-10-2025"

# ===== CREAR CARPETAS =====
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(GRAF_DIR, exist_ok=True)

# ===== FUNCIÓN DE ENTRENAMIENTO POR ZONA =====
def entrenar_por_zona(nombre_zona, path_train, path_test):
    print(f"\n🚀 Entrenando modelo para zona: {nombre_zona}")

    # Leer datasets
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    # Limpieza y coerción numérica
    for df in [train_df, test_df]:
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)

    # Separar X e y
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    # Entrenar modelo
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    model.fit(X_train, y_train)

    # Predicción y métricas
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ {nombre_zona}: MAE={mae:.3f} kB | RMSE={rmse:.3f} kB | R²={r2:.3f}")

    # Guardar predicciones
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "error_abs": np.abs(y_test - y_pred)
    })
    pred_path = os.path.join(PRED_DIR, f"{nombre_zona}_pred.csv")
    pred_df.to_csv(pred_path, index=False)

    # Graficar serie temporal
    plt.figure(figsize=(10, 5))
    plt.plot(pred_df["y_true"].values, label="Real", linewidth=2)
    plt.plot(pred_df["y_pred"].values, label="Predicho", linestyle="--", linewidth=2)
    plt.title(f"Random Forest - {nombre_zona}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, f"{nombre_zona}_serie.png"), dpi=300)
    plt.close()

    # Graficar dispersión
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["y_true"], pred_df["y_pred"], alpha=0.6, color="#007acc", label="Predicciones")
    plt.plot(
        [pred_df["y_true"].min(), pred_df["y_true"].max()],
        [pred_df["y_true"].min(), pred_df["y_true"].max()],
        color="red", linestyle="--", linewidth=2, label="Línea perfecta (y=x)"
    )
    plt.title(f"Dispersión Real vs Predicho - {nombre_zona}")
    plt.xlabel("Valor real (kB)")
    plt.ylabel("Predicción (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, f"{nombre_zona}_dispersion.png"), dpi=300)
    plt.close()

    return {"zona": nombre_zona, "MAE": mae, "RMSE": rmse, "R2": r2}

# ===== RECORRER TODAS LAS ZONAS =====
metricas = []
train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))

for path_train in train_files:
    nombre_zona = os.path.splitext(os.path.basename(path_train))[0]
    path_test = os.path.join(TEST_DIR, os.path.basename(path_train))

    if not os.path.exists(path_test):
        print(f"⚠️ No se encontró test para {nombre_zona}, se omite.")
        continue

    metrica = entrenar_por_zona(nombre_zona, path_train, path_test)
    metricas.append(metrica)

# ===== GUARDAR RESUMEN GLOBAL =====
metricas_df = pd.DataFrame(metricas)
metricas_df.to_csv(RESULTADOS_PATH, index=False)

print("\n📊 Entrenamiento completado para todas las zonas.")
print(f"📁 Resultados guardados en: {RESULTADOS_PATH}")
print(f"📂 Gráficas en: {GRAF_DIR}")
print(f"📂 Predicciones en: {PRED_DIR}")

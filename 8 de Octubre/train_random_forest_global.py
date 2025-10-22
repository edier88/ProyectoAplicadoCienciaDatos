# ===========================================================
# train_random_forest_global_v2.py
# Entrena un modelo Random Forest global sobre los datasets
# ventaneados (train_windowed/ y test_windowed/).
# Incluye validación automática y limpieza de datos numéricos.
# 07-10-2025
# ===========================================================

import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============ CONFIGURACIÓN ============
TRAIN_DIR = "train_windowed/"
TEST_DIR = "test_windowed/"
TARGET = "target"
N_ESTIMATORS = 300
RANDOM_STATE = 42
N_JOBS = -1  # usa todos los núcleos disponibles

# ============ FUNCIONES UTILITARIAS ============
def load_and_concat(path):
    """Carga todos los CSV en una carpeta y los concatena en un solo DataFrame."""
    files = glob.glob(os.path.join(path, "*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def convertir_a_numerico(df):
    """Convierte todos los valores del DataFrame a float, reemplazando comas por puntos."""
    for col in df.columns:
        # reemplazar comas decimales y forzar conversión a numérico
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validar_dataset(df, name="Dataset"):
    """
    Verifica si hay columnas con valores no numéricos o NaN.
    Imprime un resumen con alertas.
    """
    print(f"\n🔍 Validando {name}...")
    non_numeric = []
    for col in df.columns:
        # detectar columnas con valores NaN o no numéricos
        if df[col].dtype not in [np.float64, np.int64]:
            non_numeric.append(col)
        elif df[col].isna().any():
            print(f"⚠️  Columna '{col}' contiene NaN ({df[col].isna().sum()} valores).")

    if non_numeric:
        print(f"⚠️  Columnas no numéricas detectadas: {non_numeric}")
    else:
        print("✅ Todas las columnas son numéricas.")
    print(f"✅ Tamaño final de {name}: {df.shape[0]} filas × {df.shape[1]} columnas")

# ============ 1. CARGAR DATOS ============
print("Cargando datasets ventaneados...")

train_df = load_and_concat(TRAIN_DIR)
test_df = load_and_concat(TEST_DIR)

print(f"Train: {train_df.shape[0]} filas, {train_df.shape[1]} columnas")
print(f"Test : {test_df.shape[0]} filas, {test_df.shape[1]} columnas")

# ============ 2. VALIDACIÓN Y LIMPIEZA ============
print("\nLimpieza y conversión de formatos numéricos...")

train_df = convertir_a_numerico(train_df)
test_df = convertir_a_numerico(test_df)

# Reemplazar posibles NaN generados por coerción
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# Validar datasets
validar_dataset(train_df, "Train")
validar_dataset(test_df, "Test")

# Separar X e y
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ============ 3. ENTRENAMIENTO ============
print("\nEntrenando modelo Random Forest global...")

rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

rf.fit(X_train, y_train)

# ============ 4. PREDICCIÓN Y MÉTRICAS ============
print("\nEvaluando desempeño...")

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n✅ MAE  (Mean Absolute Error): {mae:,.2f} kB")
print(f"✅ RMSE (Root Mean Squared Error): {rmse:,.2f} kB")

# ============ 5. GUARDAR RESULTADOS ============
results = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
results["error_abs"] = abs(results["y_true"] - results["y_pred"])

results.to_csv("resultados_random_forest.csv", index=False)

print("\n📁 Archivo guardado: resultados_random_forest.csv")
print("Entrenamiento y evaluación completados exitosamente.")

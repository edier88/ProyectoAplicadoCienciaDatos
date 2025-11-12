# -*- coding: utf-8 -*-
"""
Script: 6.scale_with_robust_v3_hybrid.py
----------------------------------------
Objetivo:
    Escalar datasets de entrenamiento y prueba (train/test) por zona WiFi
    utilizando el RobustScaler de scikit-learn. Esta versión combina lo mejor
    de las versiones anteriores:
        - Ajusta el escalador solo con los datos de entrenamiento (buenas prácticas).
        - Aplica el mismo escalador al conjunto de prueba.
        - Genera dos escaladores por zona:
            1️⃣ Escalador general (todas las columnas numéricas).
            2️⃣ Escalador secundario (solo NUMERO.CONEXIONES, USAGE.KB, PORCENTAJE.USO).
        - Guarda los archivos escalados y los .pkl correspondientes.

Entradas:
    - train-70/  : archivos CSV de entrenamiento por zona.
    - test-30/   : archivos CSV de prueba por zona.

Salidas:
    - train_scaled/       : datasets escalados de entrenamiento.
    - test_scaled/        : datasets escalados de prueba.
    - scalers_original/   : escaladores por zona (general y 3 variables).

Autor:
    Equipo de Tesis - Maestría en Ciencia de Datos (Pamartin & Edier)
Fecha:
    2025-11-10
"""

# ======================================================
# 🔹 1. Importaciones necesarias
# ======================================================
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from joblib import dump

# ======================================================
# 🔹 2. Configuración de rutas
# ======================================================
INPUT_TRAIN = "train-70"
INPUT_TEST = "test-30"
OUTPUT_TRAIN = "train_scaled"
OUTPUT_TEST = "test_scaled"
OUTPUT_SCALERS = "scalers_original"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_TEST, exist_ok=True)
os.makedirs(OUTPUT_SCALERS, exist_ok=True)

# ======================================================
# 🔹 3. Función de escalado por zona
# ======================================================
def escalar_zona(train_path, test_path):
    """
    Escala los datasets de entrenamiento y prueba para una zona.
    Crea dos escaladores: general y reducido (3 variables).
    """

    base_name = os.path.basename(train_path)
    print(f"\n📂 Procesando zona: {base_name}")

    # --------------------------------------------------
    # 3.1 Cargar datasets
    # --------------------------------------------------
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if df_train.empty or df_test.empty:
        print(f"⚠️ {base_name}: dataset vacío. Se omite.")
        return

    # --------------------------------------------------
    # 3.2 Identificar columnas numéricas
    # --------------------------------------------------
    exclude_cols = ["DIA_SEMANA", "LABORAL", "FIN_DE_SEMANA", "FESTIVO"]
    num_cols = [
        col for col in df_train.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[col])
    ]

    if not num_cols:
        print(f"⚠️ {base_name}: no se encontraron columnas numéricas.")
        return

    # --------------------------------------------------
    # 3.3 Crear y ajustar el RobustScaler con el train
    # --------------------------------------------------
    scaler_general = RobustScaler()
    df_train[num_cols] = scaler_general.fit_transform(df_train[num_cols])

    # Aplicar el mismo escalador al test
    df_test[num_cols] = scaler_general.transform(df_test[num_cols])

    # --------------------------------------------------
    # 3.4 Guardar datasets escalados
    # --------------------------------------------------
    output_train_path = os.path.join(OUTPUT_TRAIN, base_name)
    output_test_path = os.path.join(OUTPUT_TEST, base_name)
    df_train.to_csv(output_train_path, index=False, encoding="utf-8")
    df_test.to_csv(output_test_path, index=False, encoding="utf-8")

    print(f"  ✅ Train escalado → {output_train_path}")
    print(f"  ✅ Test escalado  → {output_test_path}")

    # --------------------------------------------------
    # 3.5 Guardar escalador general
    # --------------------------------------------------
    scaler_general_path = os.path.join(OUTPUT_SCALERS, base_name.replace(".csv", ".pkl"))
    dump(scaler_general, scaler_general_path)
    print(f"  💾 Scaler general guardado → {scaler_general_path}")

    # --------------------------------------------------
    # 3.6 Escalador secundario (solo 3 variables)
    # --------------------------------------------------
    cols_target = ["NUMERO.CONEXIONES", "USAGE.KB", "PORCENTAJE.USO"]
    cols_target = [c for c in cols_target if c in df_train.columns]

    if len(cols_target) == 3:
        scaler_target = RobustScaler()
        scaler_target.fit(df_train[cols_target])

        scaler_target_name = base_name.replace(".csv", "_3vars.pkl")
        scaler_target_path = os.path.join(OUTPUT_SCALERS, scaler_target_name)
        dump(scaler_target, scaler_target_path)

        print(f"  ➕ Scaler (3 variables) guardado → {scaler_target_path}")
    else:
        print(f"  ⚠️ {base_name}: columnas faltantes para scaler secundario ({cols_target})")

    print(f"  [OK] Escalado y guardado: {base_name}")

# ======================================================
# 🔹 4. Bucle principal
# ======================================================
def main():
    """Procesa todas las zonas con datasets en train/test."""
    archivos_train = sorted([f for f in os.listdir(INPUT_TRAIN) if f.endswith(".csv")])

    if not archivos_train:
        print("⚠️ No se encontraron archivos CSV en train-70/.")
        return

    print(f"📊 Total de zonas a procesar: {len(archivos_train)}")

    for archivo in archivos_train:
        train_path = os.path.join(INPUT_TRAIN, archivo)
        test_path = os.path.join(INPUT_TEST, archivo)

        if not os.path.exists(test_path):
            print(f"⚠️ No existe archivo de test correspondiente para {archivo}. Se omite.")
            continue

        try:
            escalar_zona(train_path, test_path)
        except Exception as e:
            print(f"❌ Error procesando {archivo}: {e}")

    print("\n🏁 Proceso completado. Todos los escaladores generados correctamente.")

# ======================================================
# 🔹 5. Ejecución directa
# ======================================================
if __name__ == "__main__":
    main()

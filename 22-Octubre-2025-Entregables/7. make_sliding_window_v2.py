# make_sliding_window_v3.py
# ===========================================================
# Ventaneo extendido (versión final según reunión 2025-09-24)
# Incluye:
#   - Los 7 días previos (t-7 ... t-1) de TODOS los atributos (incluyendo USAGE.KB)
#   - Los 6 atributos del día a predecir (t0), EXCEPTO USAGE.KB
#   - Target = USAGE.KB del día a predecir (t0)
# 07-10-2025
# ===========================================================

import os
import glob
import pandas as pd

def crear_directorio(path):
    """Crea la carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def generar_ventanas_extendidas(df, target_col='USAGE.KB', window_size=7):
    """
    Genera el ventaneo extendido con todos los atributos:
    - 7 días previos (t-7 ... t-1) de TODOS los atributos (incluido USAGE.KB)
    - Los atributos del día a predecir (t0), EXCEPTO el target
    - Target = USAGE.KB del día a predecir (día N+1)
    """
    df = df.reset_index(drop=True)

    # 🔹 Eliminar FECHA.CONEXION si existe
    if 'FECHA.CONEXION' in df.columns:
        df = df.drop(columns=['FECHA.CONEXION'])

    # 🔹 Separar las columnas (todas)
    all_features = list(df.columns)
    features_no_target = [c for c in all_features if c != target_col]

    data = []

    # 🔹 Crear las ventanas
    for i in range(len(df) - window_size):
        ventana = []

        # 1️⃣ Atributos de los 7 días previos (incluye USAGE.KB)
        for lag in range(window_size):
            for f in all_features:
                ventana.append(df.loc[i + lag, f])

        # 2️⃣ Atributos del día a predecir (sin USAGE.KB)
        for f in features_no_target:
            ventana.append(df.loc[i + window_size, f])

        # 3️⃣ Target = USAGE.KB del día a predecir (día N+1)
        y = df.loc[i + window_size, target_col]

        fila = ventana + [y]
        data.append(fila)

    # 🔹 Nombres de columnas
    colnames = []
    for lag in range(window_size):
        for f in all_features:
            colnames.append(f"{f}_t-{window_size - lag}")
    for f in features_no_target:
        colnames.append(f"{f}_t0")
    colnames += ['target']

    return pd.DataFrame(data, columns=colnames)

def procesar_datasets(input_train="train_scaled/",
                      input_test="test_scaled/",
                      output_train="train_windowed/",
                      output_test="test_windowed/",
                      target_col="USAGE.KB",
                      window_size=7):
    """Procesa todos los CSV en train y test aplicando el ventaneo extendido."""

    crear_directorio(output_train)
    crear_directorio(output_test)

    # Procesar TRAIN
    for file in glob.glob(os.path.join(input_train, "*.csv")):
        base = os.path.basename(file)
        df = pd.read_csv(file)
        df_windowed = generar_ventanas_extendidas(df, target_col, window_size)
        df_windowed.to_csv(os.path.join(output_train, base), index=False)
        print(f"[OK] Train ventaneado: {base} ({df_windowed.shape[0]} filas, {df_windowed.shape[1]} columnas)")

    # Procesar TEST
    for file in glob.glob(os.path.join(input_test, "*.csv")):
        base = os.path.basename(file)
        df = pd.read_csv(file)
        df_windowed = generar_ventanas_extendidas(df, target_col, window_size)
        df_windowed.to_csv(os.path.join(output_test, base), index=False)
        print(f"[OK] Test ventaneado: {base} ({df_windowed.shape[0]} filas, {df_windowed.shape[1]} columnas)")

if __name__ == "__main__":
    procesar_datasets(
        input_train="train_scaled/",
        input_test="test_scaled/",
        output_train="train_windowed/",
        output_test="test_windowed/",
        target_col="USAGE.KB",
        window_size=7
    )

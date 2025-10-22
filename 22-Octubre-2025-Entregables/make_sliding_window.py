# make_sliding_window.py
# ===========================================================
# Genera datasets con ventana deslizante (sliding window)
# a partir de los CSV escalados en train_scaled/ y test_scaled/.
# ===========================================================

import os
import glob
import pandas as pd


def crear_directorio(path):
    """Crea la carpeta si no existe"""
    if not os.path.exists(path):
        os.makedirs(path)


def generar_ventanas(df, target_col, window_size=7):
    """
    Convierte una serie temporal en formato supervisado usando ventana deslizante.
    - df: DataFrame con columna target ya escalada
    - target_col: nombre de la columna objetivo (ej. 'USAGE.KB')
    - window_size: tamaño de la ventana (ej. 7 días)
    """
    data = []
    for i in range(len(df) - window_size):
        # variables de entrada: últimos N días
        X = df[target_col].iloc[i:i+window_size].values
        # salida: día siguiente
        y = df[target_col].iloc[i+window_size]
        fila = list(X) + [y]
        data.append(fila)

    # nombres de columnas
    cols = [f"lag{j+1}" for j in range(window_size)] + ["target"]
    return pd.DataFrame(data, columns=cols)


def procesar_datasets(input_train="train_scaled/",
                      input_test="test_scaled/",
                      output_train="train_windowed/",
                      output_test="test_windowed/",
                      target_col="USAGE.KB",
                      window_size=7):
    """Procesa todos los CSV en train y test aplicando el ventaneo"""

    crear_directorio(output_train)
    crear_directorio(output_test)

    # Procesar TRAIN
    for file in glob.glob(os.path.join(input_train, "*.csv")):
        base = os.path.basename(file)
        df = pd.read_csv(file)
        df_windowed = generar_ventanas(df, target_col, window_size)
        df_windowed.to_csv(os.path.join(output_train, base), index=False)
        print(f"[OK] Train ventaneado: {base} ({len(df_windowed)} filas)")

    # Procesar TEST
    for file in glob.glob(os.path.join(input_test, "*.csv")):
        base = os.path.basename(file)
        df = pd.read_csv(file)
        df_windowed = generar_ventanas(df, target_col, window_size)
        df_windowed.to_csv(os.path.join(output_test, base), index=False)
        print(f"[OK] Test ventaneado: {base} ({len(df_windowed)} filas)")


if __name__ == "__main__":
    # Parámetros ajustables
    procesar_datasets(
        input_train="train_scaled/",
        input_test="test_scaled/",
        output_train="train_windowed/",
        output_test="test_windowed/",
        target_col="USAGE.KB",   # aquí defines la columna objetivo
        window_size=7            # tamaño de ventana (ej. 3, 7, 30)
    )

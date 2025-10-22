# scale_with_robust.py
# 21092025 
# modificado para exluir columnas  DIA_SEMANA,  LABORAL, FIN_DE_SEMANA, FESTIVO
# ==========================================
# Script para escalar datasets de zonas Wi-Fi
# Usando RobustScaler (robusto a outliers).
# - Lee los CSV de las carpetas train-70/ y test-30/
# - Ajusta el escalador SOLO con los datos de train-70
# - Aplica la misma transformación a test-30/
# - Guarda resultados en train_scaled/ y test_scaled/
# ==========================================

import os
import glob
import pandas as pd
from sklearn.preprocessing import RobustScaler


def crear_directorio(path):
    """Crea la carpeta si no existe"""
    if not os.path.exists(path):
        os.makedirs(path)


def escalar_datasets(input_train="train-70/",
                     input_test="test-30/",
                     output_train="train_scaled/",
                     output_test="test_scaled/"):
    """
    Escala los datasets usando RobustScaler.
    Ajusta con train y aplica a train + test.
    Excluye columnas categóricas ya codificadas.
    """

    # Crear carpetas de salida
    crear_directorio(output_train)
    crear_directorio(output_test)

    # Listar archivos CSV de train y test
    train_files = glob.glob(os.path.join(input_train, "*.csv"))
    test_files = glob.glob(os.path.join(input_test, "*.csv"))

    # Columnas que NO deben escalarse
    cols_excluir = ["DIA_SEMANA", "LABORAL", "FIN_DE_SEMANA", "FESTIVO"]

    for train_file in train_files:
        # Nombre base (ej: Parque_del_Perro.csv)
        base_name = os.path.basename(train_file)

        # Ruta equivalente en test
        test_file = os.path.join(input_test, base_name)
        if not os.path.exists(test_file):
            print(f"[ADVERTENCIA] No se encontró test para {base_name}, se omite.")
            continue

        # Cargar datasets
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        # Seleccionar columnas numéricas para escalar
        num_cols = df_train.select_dtypes(include=["int64", "float64"]).columns

        # Excluir columnas categóricas indicadas
        num_cols = [c for c in num_cols if c not in cols_excluir]

        # Ajustar el escalador SOLO con train
        scaler = RobustScaler()
        df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

        # Aplicar la misma transformación a test
        df_test[num_cols] = scaler.transform(df_test[num_cols])

        # Guardar resultados
        df_train.to_csv(os.path.join(output_train, base_name), index=False)
        df_test.to_csv(os.path.join(output_test, base_name), index=False)

        print(f"[OK] Escalado y guardado: {base_name}")


if __name__ == "__main__":
    escalar_datasets()

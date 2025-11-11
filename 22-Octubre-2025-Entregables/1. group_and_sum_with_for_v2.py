# -*- coding: utf-8 -*-
"""
Script: group_and_sum_with_for_v2.py
------------------------------------------
Objetivo:
    Este script agrupa y suma los datos de tráfico WiFi por zona y fecha
    dentro de la carpeta `csv-zonas-wifi-sin-agrupar-no-sumarizado/`.
    Su propósito es consolidar múltiples registros diarios por zona,
    garantizando que los campos numéricos sean tratados correctamente
    como valores float y las variables categóricas estén normalizadas.

    Este proceso es parte del preprocesamiento del proyecto de tesis
    “Predicción del tráfico de datos en las zonas WiFi públicas de Cali”.

Autor:
    Equipo de Tesis - Maestría en Ciencia de Datos (Pamartin & Edier)
Fecha:
    2025-11-08

Cumple con:
    - Estándar PEP 8 (indentación, nombres, comentarios, docstrings)
    - Reproducibilidad de resultados
"""

# ======================================================
# 🔹 1. Importación de librerías necesarias
# ======================================================
import pandas as pd        # Manipulación y análisis de datos
from pathlib import Path   # Manejo seguro de rutas de archivos
import os                  # Operaciones del sistema (crear carpetas, etc.)

# ======================================================
# 🔹 2. Configuración de carpetas de entrada y salida
# ======================================================

# Carpeta donde se encuentran los CSV sin agrupar (dataset base)
carpeta_entrada = Path("csv-zonas-wifi-sin-agrupar-no-sumarizado")

# Carpeta donde se guardarán los CSV agrupados y sumados
carpeta_salida = Path("csv-zonas-wifi-agrupados-sumados")

# Crea la carpeta de salida si no existe (no genera error si ya está creada)
carpeta_salida.mkdir(exist_ok=True)

# ======================================================
# 🔹 3. Definición de columnas de agrupación y numéricas
# ======================================================

# Columnas categóricas utilizadas para realizar el agrupamiento
# (corresponden a las dimensiones por las cuales se agregan los datos)
columnas_agrupacion = [
    "FECHA.CONEXION",
    "AREA",
    "NOMBRE.ZONA",
    "COMUNA",
    "MODEL",
    "es_festivo",
    "tipo_dia",
    "dia_semana",
    "LATITUD",
    "LONGITUD"
]

# Columnas numéricas sobre las que se realizarán las sumas
columnas_numericas = [
    "NUMERO.CONEXIONES",
    "USAGE.KB",
    "PORCENTAJE.USO"
]

# ======================================================
# 🔹 4. Bucle principal: iterar sobre cada archivo CSV
# ======================================================

for archivo in carpeta_entrada.glob("*.csv"):

    print(f"\n📄 Procesando archivo: {archivo.name}")

    # ------------------------------------------------------
    # 4.1 Lectura del archivo CSV
    # ------------------------------------------------------
    # Se especifica `decimal=","` por si los valores usan coma como separador.
    df = pd.read_csv(archivo, decimal=",")

    # ------------------------------------------------------
    # 4.2 Conversión de columnas numéricas a tipo float
    # ------------------------------------------------------
    # Garantiza que las columnas de interés sean numéricas
    # (remueve símbolos como % o comas, y reemplaza errores con 0)
    for col in columnas_numericas:
        if col in df.columns:
            # Convertir valores a texto para poder limpiar símbolos
            df[col] = (
                df[col]
                .astype(str)                # Asegura que todo sea string
                .str.replace('%', '', regex=False)  # Elimina símbolo de porcentaje
                .str.replace(',', '.', regex=False) # Cambia coma decimal a punto
                .str.strip()                # Quita espacios en blanco
            )

            # Convierte los textos limpios a tipo numérico
            # Valores no convertibles se reemplazan por NaN, luego por 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            print(f"  ✓ Columna '{col}' convertida a numérico correctamente.")

    # ------------------------------------------------------
    # 4.3 Normalización de columnas categóricas
    # ------------------------------------------------------
    # Convierte todas las columnas categóricas a texto en mayúsculas
    # y elimina espacios antes/después. Esto evita diferencias como:
    # “Festivo” vs “ festivo” vs “FESTIVO”.
    for col in columnas_agrupacion:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # ------------------------------------------------------
    # 4.4 Agrupamiento y sumatoria
    # ------------------------------------------------------
    # Se agrupa por las columnas categóricas y se suman las numéricas.
    # `as_index=False` evita que las columnas de agrupación se vuelvan índices.
    df_agrupado = df.groupby(
        columnas_agrupacion,
        as_index=False
    ).agg({
        "NUMERO.CONEXIONES": "sum",
        "USAGE.KB": "sum",
        "PORCENTAJE.USO": "sum"
    })

    # ------------------------------------------------------
    # 4.5 Guardado del archivo agrupado
    # ------------------------------------------------------
    # Se define el nombre de salida anteponiendo el prefijo 'agrupado_'
    nombre_salida = f"agrupado_{archivo.name}"

    # Exportar el nuevo CSV con punto decimal estándar (.)
    df_agrupado.to_csv(
        carpeta_salida / nombre_salida,
        index=False,
        decimal="."
    )

    print(f"  ✅ Archivo agrupado y guardado como: {nombre_salida}")

# ======================================================
# 🔹 5. Finalización del proceso
# ======================================================
print("\n🎉 ¡Procesamiento completado con éxito!")
print(f"📂 Archivos finales guardados en: {carpeta_salida}")

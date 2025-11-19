#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 13 – Imputación robusta de días faltantes (v7.2)
Método: Mediana por DOW + Rolling Mean
Corrección: respeta EXACTAMENTE las columnas del CSV original
"""

import os
import glob
import pandas as pd
import numpy as np


# CONFIGURACIÓN
CARPETA_INPUT = "csv-zonas-wifi-1-solo-AP-ORIG-V2/"
CARPETA_OUTPUT = "csv-zonas-wifi-1-solo-AP-ORIG-V2-imputado/"

COLUMNAS_NUMERICAS = [
    "NUMERO.CONEXIONES",
    "USAGE.KB",
    "PORCENTAJE.USO"
]

COLUMNAS_FIJAS = [
    "DIA_SEMANA",
    "LABORAL",
    "FIN_DE_SEMANA",
    "FESTIVO"
]

WINDOW_ROLLING = 7
GUARDAR_REPORTE = True


# ============================================================
# IMPUTACIÓN
# ============================================================

def imputacion_mediana_dow(df, columna):
    df = df.copy()
    df["dow"] = df["FECHA.CONEXION"].dt.dayofweek
    medianas = df.groupby("dow")[columna].transform("median")
    return df[columna].fillna(medianas)


def imputacion_rolling(df, columna, window=7):
    df = df.copy()
    rolling_mean = df[columna].rolling(window=window, min_periods=1).mean()
    return df[columna].fillna(rolling_mean)


def imputacion_hibrida(df, columna):
    df[columna] = imputacion_mediana_dow(df, columna)
    df[columna] = imputacion_rolling(df, columna)
    df[columna] = df[columna].ffill()
    df[columna] = df[columna].bfill()
    return df[columna]


# ============================================================
# EXPANDIR FECHAS
# ============================================================

def expandir_fechas(df):
    df = df.copy()
    df["FECHA.CONEXION"] = pd.to_datetime(df["FECHA.CONEXION"])
    df = df.sort_values("FECHA.CONEXION")

    fecha_min = df["FECHA.CONEXION"].min()
    fecha_max = df["FECHA.CONEXION"].max()

    fechas = pd.date_range(start=fecha_min, end=fecha_max, freq="D")
    panel = pd.DataFrame({"FECHA.CONEXION": fechas})

    panel = panel.merge(df, on="FECHA.CONEXION", how="left")

    return panel


# ============================================================
# PROCESAR ARCHIVO
# ============================================================

def procesar_archivo(archivo_path, output_path):
    try:
        df = pd.read_csv(archivo_path)

        nombre_zona = os.path.basename(archivo_path).replace(".csv", "")
        print(f"\n[Procesando] {nombre_zona}")

        df_panel = expandir_fechas(df)

        # Convertir numéricas
        for col in COLUMNAS_NUMERICAS:
            df_panel[col] = (
                df_panel[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df_panel[col] = pd.to_numeric(df_panel[col], errors="coerce")

        dias_na_antes = df_panel[COLUMNAS_NUMERICAS].isna().any(axis=1).sum()

        # Imputación
        for col in COLUMNAS_NUMERICAS:
            df_panel[col] = imputacion_hibrida(df_panel, col)

        # ============================================
        # RECONSTRUIR CAMPOS CATEGÓRICOS
        # ============================================

        # 1. Rellenar todas las columnas FIJAS con su valor original
        for col in COLUMNAS_FIJAS:
            if col in df_panel.columns:
                v = df[col].dropna().iloc[0]
                df_panel[col] = df_panel[col].fillna(v)

        # 2. Reconstruir categorías de fecha
        df_panel["es_festivo"] = False

        df_panel["tipo_dia"] = df_panel["FECHA.CONEXION"].dt.dayofweek.apply(
            lambda x: "Fin de Semana" if x >= 5 else "Laboral"
        )

        df_panel["dia_semana"] = df_panel["FECHA.CONEXION"].dt.day_name(locale="es_ES")

        dias_na_despues = df_panel[COLUMNAS_NUMERICAS].isna().any(axis=1).sum()

        # Mantener SOLO tus columnas exactas
        columnas_finales = [
            "FECHA.CONEXION",
            "AREA",
            "NOMBRE.ZONA",
            "COMUNA",
            "MODEL",
            "NUMERO.CONEXIONES",
            "USAGE.KB",
            "PORCENTAJE.USO",
            "es_festivo",
            "tipo_dia",
            "dia_semana",
            "LATITUD",
            "LONGITUD"
        ]

        df_panel = df_panel[columnas_finales]

        df_panel.to_csv(output_path, index=False, encoding="utf-8")

        print(f"   NA antes: {dias_na_antes} | NA después: {dias_na_despues}")

        return {
            "zona": nombre_zona,
            "na_antes": dias_na_antes,
            "na_despues": dias_na_despues,
            "filas": len(df_panel)
        }

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None


# ============================================================
# PROCESAR TODOS
# ============================================================

def procesar_todos():
    os.makedirs(CARPETA_OUTPUT, exist_ok=True)

    print("\n===== IMPUTACIÓN v7.2 (Final Exacta) =====\n")

    archivos = sorted(glob.glob(os.path.join(CARPETA_INPUT, "*.csv")))
    resultados = []

    for archivo in archivos:
        output = os.path.join(CARPETA_OUTPUT, os.path.basename(archivo))
        r = procesar_archivo(archivo, output)
        if r:
            resultados.append(r)

    if GUARDAR_REPORTE:
        rep = pd.DataFrame(resultados)
        rep.to_csv("reporte_imputacion_v7_2.csv", index=False)
        print("\n===== REPORTE FINAL =====")
        print(rep)


if __name__ == "__main__":
    procesar_todos()

# -*- coding: utf-8 -*-
"""
Estandariza todos los CSV en resultados_agrupados/ y genera mapas de calor (Pearson).
Uso:
    python generar_mapas_calor_min.py\
Requisitos:
    pip install pandas numpy matplotlib seaborn python-dateutil
"""

import os
import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Config ----------
# este script en este punto crea dentro de salida-csv-zonas-wifi-agrupados-sumados/estandarizados/*.csv
SRC_DIR = "csv-zonas-wifi-agrupados-sumados"
OUT_CSV_DIR = os.path.join("salida-csv-zonas-wifi-agrupados-sumados", "estandarizados")
OUT_PLOT_DIR = os.path.join("salida-csv-zonas-wifi-agrupados-sumados", "mapas_calor")
os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# alias frecuentes en tus archivos
DATE_CANDIDATES = [
    "fechaconexion", "fecha_conexion", "fechaconexion", "fechaconexion", "fechadeconexion", "fecha",
    "date", "dia", "día"
]
CONN_CANDIDATES = [
    "numeroconexiones", "numero_conexiones", "numeroconexiones", "numconexiones", "conexiones",
    "nroconexiones", "connections"
]
USAGE_KB_CANDIDATES = ["usagekb", "usage_kb", "kb", "traficokb", "usokb"]
USAGE_MB_CANDIDATES = ["usagemb", "usage_mb", "mb", "usomb", "traficomb"]
PCT_CANDIDATES = [
    "porcentajeuso", "porcentaje_uso", "porcentajeuso", "pctuso", "porcuso", "usagepercent",
]

# ---------- Utilidades ----------
def normalize_cols(cols):
    """Minúsculas, sin tildes ni símbolos; espacios -> _"""
    import unicodedata
    out = []
    for c in cols:
        c2 = c.strip().lower()
        c2 = "".join(ch for ch in unicodedata.normalize("NFKD", c2) if not unicodedata.combining(ch))
        c2 = re.sub(r"[^a-z0-9\s_/()%]", "", c2)
        c2 = c2.replace("/", " ").replace("\\", " ")
        c2 = re.sub(r"\s+", " ", c2).strip().replace(" ", "_")
        out.append(c2)
    return out

def first_match(df_cols, candidates):
    s = set(df_cols)
    for cand in candidates:
        if cand in s:
            return cand
    return None

def to_number(series):
    # convierte strings con %, comas, puntos, etc. a número
    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce",
    )

def ensure_daily_continuity(df, date_col, numeric_cols):
    df = df.sort_values(date_col)
    if df.empty:
        return df
    start, end = df[date_col].min(), df[date_col].max()
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.set_index(date_col).reindex(full_idx)
    df.index.name = date_col
    # llena NaN en numéricos con 0; deja otros como NaN
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df.reset_index().rename(columns={"index": date_col})

def summarize_by_day(df, date_col, numeric_cols):
    # si vienen dos APs el mismo día, sumamos
    agg = {c: "sum" for c in numeric_cols if c in df.columns}
    out = df.groupby(date_col, as_index=False).agg(agg)
    return out

def make_heatmap(corr, title, out_path):
    # Diccionario para mapear nombres normalizados a nombres originales
    var_map = {
        "numeroconexiones": "NUMERO.CONEXIONES",
        "porcentaje_uso_std": "PORCENTAJE.USO",
        "usage_mb_std": "USAGE.KB"
    }
    # Renombrar filas y columnas del dataframe de correlación
    corr = corr.rename(index=var_map, columns=var_map)
    # Reordenar filas y columnas según el formato solicitado
    orden = ["NUMERO.CONEXIONES", "PORCENTAJE.USO", "USAGE.KB"]
    corr = corr.loc[orden, orden]
    plt.figure(figsize=(6, 5), dpi=160)
    sns.heatmap(
        corr,
        vmin=-1, vmax=1, annot=True, fmt=".2f",
        cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- Proceso principal ----------
def process_csv(path):
    name = os.path.basename(path)
    print(f"Procesando: {name}")

    # lectura robusta (detecta ; o ,)
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")

    # normaliza nombres
    df.columns = normalize_cols(df.columns)

    # detecta columnas
    conn_col = first_match(df.columns, CONN_CANDIDATES)
    usage_kb_col = first_match(df.columns, USAGE_KB_CANDIDATES)
    pct_col = first_match(df.columns, PCT_CANDIDATES)

    # convierte numéricos
    if conn_col:
        df[conn_col] = to_number(df[conn_col]).fillna(0).astype(float)
    if usage_kb_col:
        df[usage_kb_col] = to_number(df[usage_kb_col])
    if pct_col:
        df[pct_col] = to_number(df[pct_col])

    # selecciona columnas numéricas originales para trabajar
    numeric_cols = []
    if conn_col: numeric_cols.append(conn_col)
    if usage_kb_col: numeric_cols.append(usage_kb_col)
    if pct_col: numeric_cols.append(pct_col)

    # Guardar el DataFrame estandarizado
    base_name, _ = os.path.splitext(name)
    output_csv_path = os.path.join(OUT_CSV_DIR, f"{base_name}_estandarizado.csv")
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"   ✔ CSV estandarizado guardado: {output_csv_path}")

    # quita columnas sin varianza (todo igual o NaN)
    num_df = df[numeric_cols].copy()
    keep = [c for c in num_df.columns if num_df[c].nunique(dropna=True) > 1]
    num_df = num_df[keep]
    if len(num_df.columns) >= 2:
        corr = num_df.corr(method="pearson")
        base, _ = os.path.splitext(name)
        out_plot = os.path.join(OUT_PLOT_DIR, f"{base}-pearson.png")
        title = f"Mapa de Calor Correlación - {base}"
        # Diccionario para mapear nombres normalizados a nombres originales
        var_map = {
            conn_col: "NUMERO.CONEXIONES",
            pct_col: "PORCENTAJE.USO",
            usage_kb_col: "USAGE.KB"
        }
        corr = corr.rename(index=var_map, columns=var_map)
        orden = ["NUMERO.CONEXIONES", "PORCENTAJE.USO", "USAGE.KB"]
        corr = corr.loc[[v for v in orden if v in corr.index], [v for v in orden if v in corr.columns]]
        plt.figure(figsize=(6, 5), dpi=160)
        sns.heatmap(
            corr,
            vmin=-1, vmax=1, annot=True, fmt=".2f",
            cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8}
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_plot)
        plt.close()
        print(f"   ✔ Heatmap: {out_plot}")
    else:
        print("   ⚠ No hay suficientes columnas numéricas con varianza para correlación.")

def main():
    if not os.path.isdir(SRC_DIR):
        raise SystemExit(f"No existe el directorio '{SRC_DIR}'")

    files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(".csv")]
    print(f"Se encontraron {len(files)} archivos CSV.")
    for f in files:
        try:
            process_csv(os.path.join(SRC_DIR, f))
        except Exception as e:
            print(f"Error procesando {f}: {e}")

if __name__ == "__main__":
    main()

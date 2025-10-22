#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# split_70_30.py 06092025 
# El train cubre del 1 de enero de 2024 hasta el 17 de noviembre de 2024, y el test toma del 
# 18 de noviembre de 2024 en adelante.
# Así garantizamos que el modelo aprende del pasado y se evalúa en el futuro inmediato de 
# la misma zona, siguiendo la regla de oro de predicción temporal.
# Esto requiere un conjunto de datasets ya normalizados, un registro 
# por dia, variables limpias, y si se tienen metadatos como es_festivo, tipo_dia, dia_semana.
"""
Split 70/30 (time-based) for daily-normalized WiFi zone datasets.

- Reads all .csv files from `resultados_agrupados/` (each one should be 1 row per day).
- Sorts by FECHA.CONEXION (ascending).
- Splits chronologically: first 70% -> train, remaining 30% -> test.
- Writes outputs to `train-70/` and `test-30/` using the same base filename
  with suffixes `_train.csv` and `_test.csv`.
- Preserves ALL original columns (no aggregation).

Usage (defaults are fine):
    python split_70_30.py
or with custom paths:
    python split_70_30.py --input resultados_agrupados --out-train train-70 --out-test test-30

Notes:
- If duplicate dates are found, you can optionally aggregate them (sum numeric columns,
  keep first non-numeric) by passing the flag `--aggregate-on-duplicates`. By default,
  the script assumes the files are already normalized and will only warn if duplicates appear.
"""

import argparse
import os
import sys
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Time-based 70/30 split for daily WiFi datasets.")
    ap.add_argument("--input", "-i", default="resultados_agrupados", help="Input folder with normalized CSVs (default: resultados_agrupados)")
    ap.add_argument("--out-train", "-tr", default="train-70", help="Output folder for train CSVs (default: train-70)")
    ap.add_argument("--out-test", "-te", default="test-30", help="Output folder for test CSVs (default: test-30)")
    ap.add_argument("--date-col", "-d", default="FECHA.CONEXION", help="Date column name (default: FECHA.CONEXION)")
    ap.add_argument("--aggregate-on-duplicates", action="store_true",
                    help="If duplicates by date exist, aggregate by date (sum numeric, first non-numeric).")
    return ap.parse_args()

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def read_csv_safely(path):
    # Try utf-8-sig first, then fallback to latin-1
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def aggregate_if_needed(df, date_col):
    # Aggregate duplicates: sum numeric columns, take first non-numeric
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude the date column from numeric sums accidentally
    numeric_cols = [c for c in numeric_cols if c != date_col]
    agg_dict = {col: "first" for col in df.columns if col != date_col}
    for col in numeric_cols:
        agg_dict[col] = "sum"
    # If a column is both in numeric and already set to 'first', 'sum' will override
    grouped = df.groupby(date_col, as_index=False).agg(agg_dict)
    return grouped

def process_file(file_path, out_train, out_test, date_col, aggregate_on_duplicates=False):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        df = read_csv_safely(file_path)
    except Exception as e:
        print(f"❌ Error leyendo {file_path}: {e}", file=sys.stderr)
        return

    if date_col not in df.columns:
        print(f"⚠️  {file_path}: no tiene la columna de fecha '{date_col}'. Se omite.", file=sys.stderr)
        return

    # Parsear fecha
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[date_col]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"ℹ️  {base_name}: {dropped} filas descartadas por fecha inválida.")

    # Duplicados por fecha
    dup_count = df.duplicated(subset=[date_col]).sum()
    if dup_count > 0:
        msg = f"⚠️  {base_name}: se detectaron {dup_count} duplicados por fecha."
        if aggregate_on_duplicates:
            print(msg + " Se agregará por fecha (sum numéricas, first no-numéricas).")
            df = aggregate_if_needed(df, date_col)
        else:
            print(msg + " No se agregará (se conservarán filas).")

    # Ordenar por fecha
    df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    if n < 2:
        print(f"⚠️  {base_name}: muy pocas filas ({n}). Se omite split.", file=sys.stderr)
        return

    train_size = int(0.7 * n)
    if train_size == 0 or train_size == n:
        print(f"⚠️  {base_name}: distribución no válida para split 70/30 (n={n}). Se omite.", file=sys.stderr)
        return

    df_train = df.iloc[:train_size].copy()
    df_test  = df.iloc[train_size:].copy()

    out_train_path = os.path.join(out_train, f"{base_name}_train.csv")
    out_test_path  = os.path.join(out_test,  f"{base_name}_test.csv")

    # Guardar
    df_train.to_csv(out_train_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(out_test_path, index=False, encoding="utf-8-sig")

    # Rango de fechas para resumen
    f0, f1 = df_train[date_col].min(), df_train[date_col].max()
    g0, g1 = df_test[date_col].min(), df_test[date_col].max()

    print(f"✅ {base_name}: {len(df_train)} train [{f0.date()} → {f1.date()}] | {len(df_test)} test [{g0.date()} → {g1.date()}]")

def main():
    args = parse_args()
    ensure_dirs(args.out_train, args.out_test)

    if not os.path.isdir(args.input):
        print(f"❌ Carpeta de entrada no existe: {args.input}", file=sys.stderr)
        sys.exit(1)

    csv_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"⚠️  No se encontraron .csv en {args.input}")
        sys.exit(0)

    print(f"🔎 Procesando {len(csv_files)} archivos desde '{args.input}' → train: '{args.out_train}', test: '{args.out_test}'")

    for file_path in sorted(csv_files):
        process_file(
            file_path=file_path,
            out_train=args.out_train,
            out_test=args.out_test,
            date_col=args.date_col,
            aggregate_on_duplicates=args.aggregate_on_duplicates
        )

if __name__ == "__main__":
    main()

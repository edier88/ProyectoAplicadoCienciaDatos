#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar mapas de calor (correlación y/o temporal) a partir de CSVs ya agrupados.
- Lee todos los CSV de una carpeta (recursivo opcional).
- Estandariza columnas según parámetros.
- (Opcional) Rellena días faltantes con 0 en el rango [fecha_min, fecha_max].
- Genera:
    * Heatmap de correlación por grupo (Zona y opcionalmente AP).
    * (Opcional) Heatmap temporal DíaSemana × Mes para una métrica seleccionada.
Salida: PNGs por archivo y un resumen .csv

Uso rápido:
python generar_mapas_calor.py \
  --input-dir resultados_agrupados \
  --out-dir mapas_calor \
  --date-col FECHA \
  --zone-col "NOMBRE ZONA" \
  --conn-col "NUMERO CONEXIONES" \
  --usage-col "USAGE KB" \
  --pct-col "PORCENTAJE USO"

Con separación por AP (si el CSV tiene columna de AP):
  --ap-col "AP_ID"

Incluir mapa temporal usando USAGE_MB promedio:
  --temporal --temporal-metric USAGE_MB

Rellenar días faltantes (0) en el rango [min,max]:
  --fill-missing
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- Utilidades -------------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s\-]+", "", str(text))
    s = re.sub(r"\s+", "_", s.strip())
    return s[:150]

def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def normalize_numeric(series: pd.Series) -> pd.Series:
    def to_num(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(",", ".")
        s = re.sub(r"[^0-9\.\-\+eE]", "", s)
        if s in {"", ".", "-", "+", "e", "E"}:
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    return series.map(to_num)

def normalize_pct(series: pd.Series) -> pd.Series:
    def to_pct(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(",", ".")
        s = re.sub(r"[^0-9\.\-\+eE]", "", s)
        if s in {"", ".", "-", "+", "e", "E"}:
            return np.nan
        try:
            v = float(s)
            if v > 1.0:  # asumir 0-100
                v = v / 100.0
            return v
        except Exception:
            return np.nan
    return series.map(to_pct)

def parse_dates(series: pd.Series) -> pd.Series:
    # dayfirst=True para formatos DD/MM/YYYY
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.normalize()

def fill_missing_days(df: pd.DataFrame, date_col: str, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    """
    Para cada grupo, reindexa por rango de fechas continuo y rellena NaN con 0 en value_cols.
    """
    out = []
    for _, g in df.groupby(group_cols):
        if g[date_col].isna().all():
            out.append(g)
            continue
        dmin, dmax = g[date_col].min(), g[date_col].max()
        idx = pd.date_range(dmin, dmax, freq="D", name=date_col)
        g2 = (g.set_index(date_col)
                .reindex(idx)
                .reset_index()
                .rename(columns={"index": date_col}))
        for c in group_cols:
            g2[c] = g[c].iloc[0]
        for c in value_cols:
            if c in g2.columns:
                g2[c] = g2[c].fillna(0.0)
        out.append(g2)
    return pd.concat(out, ignore_index=True)

def corr_heatmap(df: pd.DataFrame, cols: List[str], title: str, outfile: Path, dpi: int = 150) -> bool:
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2 or df[cols].dropna().shape[0] < 5:
        return False
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(4 + len(cols), 4 + len(cols)))
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right"); ax.set_yticklabels(cols)

    # Anotar valores
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center")

    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return True

def weekday_month_heatmap(df: pd.DataFrame, date_col: str, metric: str, title: str, outfile: Path, dpi: int = 150) -> bool:
    if metric not in df.columns or date_col not in df.columns:
        return False
    tmp = df.copy()
    tmp["MES"] = tmp[date_col].dt.month
    tmp["DIA_SEMANA"] = tmp[date_col].dt.dayofweek
    piv = tmp.pivot_table(index="DIA_SEMANA", columns="MES", values=metric, aggfunc="mean")
    if piv.empty:
        return False

    # Orden amigable
    piv = piv.reindex(index=sorted(piv.index), columns=sorted(piv.columns))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_yticks(range(len(piv.index))); ax.set_xticks(range(len(piv.columns)))
    ax.set_yticklabels(["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"][:len(piv.index)])
    ax.set_xticklabels([str(m) for m in piv.columns])
    ax.set_xlabel("Mes"); ax.set_ylabel("Día de la semana")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return True

# ------------- Pipeline -------------

def process_folder(
    input_dir: Path,
    out_dir: Path,
    recursive: bool,
    date_col: str,
    zone_col: str,
    conn_col: Optional[str],
    usage_col: Optional[str],
    pct_col: Optional[str],
    ap_col: Optional[str],
    temporal: bool,
    temporal_metric: str,
    fill_missing: bool,
    min_rows: int,
    dpi: int
):
    safe_mkdir(out_dir)
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(input_dir.glob(pattern))

    # Resumen
    summary_rows = []

    for f in files:
        try:
            raw = read_csv_any(f)
        except Exception as e:
            summary_rows.append({"archivo": str(f), "status": f"ERROR lectura: {e}"})
            continue

        cols_map = {}
        for name, std in [(date_col,"FECHA"), (zone_col,"ZONA"),
                          (conn_col,"CONEXIONES"), (usage_col,"USAGE_KB"),
                          (pct_col,"PORCENTAJE"), (ap_col,"AP")]:
            if name:
                cols_map[name] = std

        df = raw.rename(columns=cols_map)

        # Validar columnas mínimas
        if "FECHA" not in df.columns or "ZONA" not in df.columns:
            summary_rows.append({"archivo": str(f), "status": "ERROR columnas mínimas faltantes"})
            continue

        # Parsear
        df["FECHA"] = parse_dates(df["FECHA"])
        if "CONEXIONES" in df.columns: df["CONEXIONES"] = normalize_numeric(df["CONEXIONES"])
        if "USAGE_KB" in df.columns:
            df["USAGE_KB"] = normalize_numeric(df["USAGE_KB"])
            df["USAGE_MB"] = df["USAGE_KB"] / 1024.0
        if "PORCENTAJE" in df.columns: df["PORCENTAJE"] = normalize_pct(df["PORCENTAJE"])

        # Drop filas sin fecha o zona
        df = df.dropna(subset=["FECHA","ZONA"])

        # Reagrupar por día (por si el agrupado previo no consolidó por día)
        group_cols = ["ZONA","FECHA"] + (["AP"] if ap_col and "AP" in df.columns else [])
        agg_dict = {}
        if "CONEXIONES" in df.columns: agg_dict["CONEXIONES"] = "sum"
        if "USAGE_MB" in df.columns:   agg_dict["USAGE_MB"] = "sum"
        if "PORCENTAJE" in df.columns: agg_dict["PORCENTAJE"] = "mean"
        if agg_dict:
            df = (df.groupby(group_cols, dropna=False)
                    .agg(agg_dict)
                    .reset_index())

        # Relleno de días faltantes (opcional)
        if fill_missing:
            value_cols = [c for c in ["CONEXIONES","USAGE_MB","PORCENTAJE"] if c in df.columns]
            df = fill_missing_days(df, "FECHA", [c for c in ["ZONA","AP"] if c in df.columns], value_cols)

        # Generar por grupo (Zona[, AP])
        if "AP" in df.columns:
            groups = df.groupby(["ZONA","AP"])
        else:
            groups = df.groupby(["ZONA"])

        out_subdir = out_dir / slugify(f.stem)
        safe_mkdir(out_subdir)

        gen_corr = 0
        gen_temp = 0
        for gkey, gdf in groups:
            if gdf.shape[0] < min_rows:
                continue

            if isinstance(gkey, tuple):
                z = gkey[0]; ap = gkey[1] if len(gkey) > 1 else None
            else:
                z = gkey; ap = None

            title_suffix = f"Zona: {z}" + (f" | AP: {ap}" if ap is not None else "")
            slug = slugify(f"{z}__AP_{ap}") if ap is not None else slugify(f"{z}")

            # Heatmap de correlación
            cols_corr = [c for c in ["CONEXIONES","USAGE_MB","PORCENTAJE"] if c in gdf.columns]
            ok_corr = corr_heatmap(
                gdf, cols_corr,
                f"Correlación - {title_suffix} (n={gdf.shape[0]})",
                out_subdir / f"heatmap_correlacion_{slug}.png",
                dpi=dpi
            )
            if ok_corr: gen_corr += 1

            # Heatmap temporal (opcional)
            if temporal and temporal_metric in gdf.columns:
                ok_temp = weekday_month_heatmap(
                    gdf, "FECHA", temporal_metric,
                    f"Heatmap {temporal_metric} (DíaSemana × Mes) - {title_suffix}",
                    out_subdir / f"heatmap_temporal_{temporal_metric}_{slug}.png",
                    dpi=dpi
                )
                if ok_temp: gen_temp += 1

        summary_rows.append({
            "archivo": str(f),
            "status": "OK",
            "fig_corr_generadas": gen_corr,
            "fig_temporales_generadas": gen_temp
        })

    # Guardar resumen
    pd.DataFrame(summary_rows).to_csv(out_dir / "_resumen_ejecucion.csv", index=False)


def main():
    p = argparse.ArgumentParser(description="Generar mapas de calor (correlación y temporal) desde CSVs agrupados.")
    p.add_argument("--input-dir", required=True, help="Carpeta con CSVs agrupados (p. ej., resultados_agrupados/)")
    p.add_argument("--out-dir", required=True, help="Carpeta de salida para PNGs y resumen")
    p.add_argument("--recursive", action="store_true", help="Buscar CSVs de forma recursiva")
    p.add_argument("--date-col", default="FECHA", help="Nombre de la columna de fecha en los CSV")
    p.add_argument("--zone-col", default="NOMBRE ZONA", help="Nombre de la columna de zona")
    p.add_argument("--conn-col", default="NUMERO CONEXIONES", help="Columna de número de conexiones")
    p.add_argument("--usage-col", default="USAGE KB", help="Columna de uso en KB (se convierte a MB)")
    p.add_argument("--pct-col", default="PORCENTAJE USO", help="Columna de porcentaje de uso")
    p.add_argument("--ap-col", default=None, help="(Opcional) Columna de identificador de AP para separar por AP")
    p.add_argument("--temporal", action="store_true", help="Generar heatmap temporal DíaSemana × Mes")
    p.add_argument("--temporal-metric", default="USAGE_MB", help="Métrica para heatmap temporal (ej: USAGE_MB, CONEXIONES, PORCENTAJE)")
    p.add_argument("--fill-missing", action="store_true", help="Rellenar días faltantes con 0 dentro del rango de fechas por grupo")
    p.add_argument("--min-rows", type=int, default=10, help="Mínimo de filas por grupo para graficar")
    p.add_argument("--dpi", type=int, default=150, help="Resolución de salida (DPI)")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    process_folder(
        input_dir=input_dir,
        out_dir=out_dir,
        recursive=args.recursive,
        date_col=args.date_col,
        zone_col=args.zone_col,
        conn_col=args.conn_col,
        usage_col=args.usage_col,
        pct_col=args.pct_col,
        ap_col=args.ap_col,
        temporal=args.temporal,
        temporal_metric=args.temporal_metric,
        fill_missing=args.fill_missing,
        min_rows=args.min_rows,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_columns_to_uppercase.py (MEJORADO)
------------------------------------------
Convierte:
 ✓ Todos los nombres de columnas a MAYÚSCULAS
 ✓ Los valores de ES_FESTIVO, TIPO_DIA y DIA_SEMANA a MAYÚSCULAS

Uso:
    python convert_columns_to_uppercase.py
    python convert_columns_to_uppercase.py --input carpeta
"""

import os
import argparse
import sys
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convierte los nombres de columnas y ciertos valores a mayúsculas."
    )
    parser.add_argument(
        "--input", "-i", 
        default="csv-zonas-wifi-separados-PruebaEdier/",
        help="Carpeta de entrada con archivos CSV"
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Crear copia de respaldo"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Mostrar cambios sin aplicar"
    )
    return parser.parse_args()


def list_csvs(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv")
    ])


def process_csv_file(file_path, backup=False, dry_run=False):
    filename = os.path.basename(file_path)
    print(f"\n🔧 Procesando: {filename}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

        old_columns = df.columns.tolist()
        new_columns = [c.upper() for c in old_columns]

        # Convertir nombres de columnas
        df.columns = new_columns

        # ---------------------------------------------
        # CONVERTIR A MAYÚSCULAS LOS VALORES NECESARIOS
        # ---------------------------------------------
        columnas_objetivo = ["ES_FESTIVO", "TIPO_DIA", "DIA_SEMANA"]

        for col in columnas_objetivo:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper()

        # Mostrar cambios en DRY RUN
        if dry_run:
            print("   [DRY RUN] Nuevos nombres de columnas:", new_columns)
            print("   [DRY RUN] Valores ES_FESTIVO / TIPO_DIA / DIA_SEMANA se convertirían a MAYÚSCULAS")
            return True

        # Backup
        if backup:
            backup_path = file_path.replace(".csv", "_backup.csv")
            df.to_csv(backup_path, index=False, encoding="utf-8-sig")
            print(f"   💾 Respaldo creado: {os.path.basename(backup_path)}")

        # Guardar archivo procesado
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"   ✅ Archivo actualizado: {filename}")

        return True

    except Exception as e:
        print(f"   ❌ Error procesando {filename}: {e}")
        return False


def main():
    args = parse_args()
    if not os.path.isdir(args.input):
        print(f"❌ No existe la carpeta: {args.input}", file=sys.stderr)
        sys.exit(1)

    csv_files = list_csvs(args.input)
    if not csv_files:
        print("⚠️ No se encontraron archivos CSV.")
        sys.exit(0)

    print(f"📁 Se encontraron {len(csv_files)} archivos CSV")

    processed_count = 0
    for file_path in csv_files:
        if process_csv_file(file_path, args.backup, args.dry_run):
            processed_count += 1

    print("\n🎉 ¡Proceso completado!")
    print(f"✅ Archivos procesados exitosamente: {processed_count}")


if __name__ == "__main__":
    main()

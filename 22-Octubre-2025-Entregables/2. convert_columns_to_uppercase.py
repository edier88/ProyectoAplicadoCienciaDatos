#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_columns_to_uppercase.py (ajustado)
------------------------------------------
Convierte todos los nombres de columnas a mayúsculas en archivos CSV.

Uso:
    python convert_columns_to_uppercase.py
    python convert_columns_to_uppercase.py --input csv-zonas-wifi-agrupados-sumados
"""

import os
import argparse
import sys
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convierte todos los nombres de columnas a mayúsculas en archivos CSV."
    )
    parser.add_argument(
        "--input", "-i", 
        default="csv-zonas-wifi-agrupados-sumados",
        help="Carpeta de entrada con archivos CSV (default: csv-zonas-wifi-agrupados-sumados)"
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Crear copias de respaldo antes de modificar los archivos"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Mostrar qué cambios se harían sin modificar los archivos"
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

        # Guardar los nombres originales
        old_columns = df.columns.tolist()
        new_columns = [c.upper() for c in old_columns]

        if old_columns == new_columns:
            print("   ℹ️  Los nombres de columnas ya están en mayúsculas")
            return False

        df.columns = new_columns
        print(f"   ✓ Columnas renombradas: {len(old_columns)}")

        if dry_run:
            print(f"   [DRY RUN] Se convertirían columnas: {old_columns} → {new_columns}")
            return True

        if backup:
            backup_path = file_path.replace(".csv", "_backup.csv")
            df.to_csv(backup_path, index=False, encoding="utf-8-sig")
            print(f"   💾 Respaldo creado: {os.path.basename(backup_path)}")

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
        print("⚠️  No se encontraron archivos CSV en la carpeta.")
        sys.exit(0)

    print(f"📁 Se encontraron {len(csv_files)} archivos CSV")

    processed_count = 0
    for file_path in csv_files:
        if process_csv_file(file_path, args.backup, args.dry_run):
            processed_count += 1

    print(f"\n🎉 Proceso completado!")
    if args.dry_run:
        print(f"🔍 [DRY RUN] Se procesarían {processed_count} archivos")
    else:
        print(f"✅ Se procesaron {processed_count} archivos exitosamente")

if __name__ == "__main__":
    main()

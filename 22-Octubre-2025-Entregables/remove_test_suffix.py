#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_test_suffix.py
---------------------
Script para remover el sufijo "-test" de todos los archivos CSV en una carpeta.

Funcionalidad:
- Lee todos los archivos .csv de la carpeta especificada
- Remueve el sufijo "-test" del nombre de cada archivo
- Renombra los archivos con el nuevo nombre

Uso:
    python remove_test_suffix.py
    python remove_test_suffix.py --input csv-zonas-wifi-agrupados-sumados
"""

import os
import argparse
import sys
from pathlib import Path


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Remueve el sufijo '-test' de archivos CSV en una carpeta."
    )
    parser.add_argument(
        "--input", "-i", 
        default="csv-zonas-wifi-agrupados-sumados",
        help="Carpeta de entrada con archivos CSV (default: csv-zonas-wifi-agrupados-sumados)"
    )
    parser.add_argument(
        "--ext", "-e",
        default=".csv",
        help="Extensión de archivo (default: .csv)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Mostrar qué archivos serían renombrados sin hacer cambios reales"
    )
    return parser.parse_args()


def ensure_dir(path):
    """Verifica que la carpeta exista."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No existe la carpeta: {path}")


def list_csvs(folder, ext):
    """Lista todos los archivos con la extensión especificada en la carpeta."""
    csv_files = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(ext.lower()):
            csv_files.append(os.path.join(folder, filename))
    return sorted(csv_files)


def remove_test_suffix(filename):
    """
    Remueve el sufijo '-test' del nombre del archivo.
    Ejemplo: 'archivo-test.csv' -> 'archivo.csv'
    """
    base_name = os.path.splitext(filename)[0]  # Nombre sin extensión
    extension = os.path.splitext(filename)[1]  # Solo la extensión
    
    # Remover '-test' del final del nombre base
    if base_name.endswith('-test'):
        new_base = base_name[:-5]  # Remover los últimos 5 caracteres ('-test')
        return new_base + extension
    return filename  # No hay cambios si no termina en '-test'


def safe_rename(old_path, new_path, dry_run=False):
    """
    Renombra un archivo de manera segura, evitando colisiones.
    """
    if old_path == new_path:
        return False  # No hay cambios necesarios
    
    # Verificar si el archivo destino ya existe
    if os.path.exists(new_path):
        # Generar un nombre único agregando un número
        base, ext = os.path.splitext(new_path)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        new_path = f"{base}_{counter}{ext}"
        print(f"   ⚠️  Archivo destino existe, usando: {os.path.basename(new_path)}")
    
    if dry_run:
        print(f"   [DRY RUN] {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
        return True
    else:
        try:
            os.rename(old_path, new_path)
            print(f"   ✓ {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
            return True
        except Exception as e:
            print(f"   ❌ Error renombrando {os.path.basename(old_path)}: {e}")
            return False


def process_folder(folder, ext, dry_run=False):
    """Procesa todos los archivos CSV en la carpeta especificada."""
    print(f"🔧 Procesando carpeta: {folder}")
    
    csv_files = list_csvs(folder, ext)
    if not csv_files:
        print(f"   ⚠️  No se encontraron archivos {ext} en la carpeta.")
        return 0
    
    print(f"   📁 Se encontraron {len(csv_files)} archivos {ext}")
    
    renamed_count = 0
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        new_filename = remove_test_suffix(filename)
        
        if filename != new_filename:
            new_path = os.path.join(folder, new_filename)
            if safe_rename(file_path, new_path, dry_run):
                renamed_count += 1
        else:
            print(f"   • {filename} (sin cambios)")
    
    return renamed_count


def main():
    """Función principal."""
    args = parse_args()
    
    try:
        ensure_dir(args.input)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"🎯 {'[MODO DRY RUN]' if args.dry_run else '[MODO REAL]'}")
    print(f"📂 Carpeta: {args.input}")
    print(f"📄 Extensión: {args.ext}")
    print()
    
    renamed_count = process_folder(args.input, args.ext, args.dry_run)
    
    print()
    if args.dry_run:
        print(f"🔍 [DRY RUN] Se renombrarían {renamed_count} archivos")
        print("   Para aplicar los cambios, ejecuta sin --dry-run")
    else:
        print(f"✅ Proceso completado. Se renombraron {renamed_count} archivos")


if __name__ == "__main__":
    main()

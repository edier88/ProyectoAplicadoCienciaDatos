#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_filenames.py
----------------------
Normaliza los nombres de archivos CSV en dos carpetas (train-70/ y test-30/) para
que queden sin sufijos de split, por ejemplo:
    "agrupado_001_ZW Parque Ingenio-test_train.csv"  ->  "agrupado_001_ZW Parque Ingenio.csv"
    "agrupado_001_ZW Parque Ingenio-test_test.csv"   ->  "agrupado_001_ZW Parque Ingenio.csv"

Reglas:
- Se eliminan sufijos de split antes de la extensión: _train, -train, _test, -test,
  _test_train, -test_train, _test_test, -test_test (ignorando mayúsculas/minúsculas).
- Si el nombre destino ya existe en la carpeta, se añade un sufijo incremental
  " (1)", " (2)", etc., para evitar colisiones.

Uso básico:
    python normalize_filenames.py
Opciones:
    python normalize_filenames.py --train-dir train-70 --test-dir test-30 --ext .csv

Comentarios y estilo siguiendo PEP8, en español.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable


# Patrones de sufijos a eliminar (antes de la extensión)
SUFFIX_PATTERNS = [
    r"_test_train$", r"-test_train$",
    r"_test_test$",  r"-test_test$",
    r"_train$",      r"-train$",
    r"_test$",       r"-test$",
]


def parse_args() -> argparse.Namespace:
    """Parsea argumentos CLI."""
    parser = argparse.ArgumentParser(description="Normaliza nombres de archivos CSV en train-70/ y test-30/.")
    parser.add_argument("--train-dir", default="train-70", help="Carpeta con archivos de entrenamiento (default: train-70)")
    parser.add_argument("--test-dir", default="test-30", help="Carpeta con archivos de prueba (default: test-30)")
    parser.add_argument("--ext", default=".csv", help="Extensión objetivo (default: .csv)")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    """Verifica que la carpeta exista."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No existe la carpeta: {path}")


def list_csvs(folder: str, ext: str) -> Iterable[str]:
    """Lista archivos con la extensión dada dentro de la carpeta."""
    for name in os.listdir(folder):
        if name.lower().endswith(ext.lower()):
            yield os.path.join(folder, name)


def strip_suffixes(base_without_ext: str) -> str:
    """Elimina sufijos de split definidos en SUFFIX_PATTERNS del nombre base (sin extensión)."""
    clean = base_without_ext
    for pat in SUFFIX_PATTERNS:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE)
    return clean.strip()


def safe_destination_path(folder: str, desired_name: str, ext: str) -> str:
    """
    Genera una ruta de destino que no colisione:
    - Si "desired_name.ext" existe, intenta "desired_name (1).ext", "desired_name (2).ext", etc.
    """
    candidate = os.path.join(folder, desired_name + ext)
    if not os.path.exists(candidate):
        return candidate
    # Si ya existe, iteramos con sufijos numéricos
    i = 1
    while True:
        candidate = os.path.join(folder, f"{desired_name} ({i}){ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def normalize_folder(folder: str, ext: str) -> None:
    """Normaliza todos los archivos con la extensión dada en la carpeta especificada."""
    print(f"🔧 Normalizando en: {folder}")
    for path in sorted(list_csvs(folder, ext)):
        dirname, filename = os.path.split(path)
        base, _ = os.path.splitext(filename)

        # Limpiar sufijos
        clean_base = strip_suffixes(base)

        # Evitar renombrados nulos o redundantes
        if clean_base == base:
            print(f"   • {filename}  →  (sin cambios)")
            continue

        # Construir destino seguro (evitar colisiones)
        dest_path = safe_destination_path(dirname, clean_base, ext)

        # Renombrar
        os.rename(path, dest_path)
        print(f"   ✓ {filename}  →  {os.path.basename(dest_path)}")
    print(f"✅ Carpeta lista: {folder}\n")


def main() -> None:
    """Punto de entrada principal del script."""
    args = parse_args()

    try:
        ensure_dir(args.train_dir)
        ensure_dir(args.test_dir)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)

    # Normalizamos primero train y luego test
    normalize_folder(args.train_dir, args.ext)
    normalize_folder(args.test_dir, args.ext)

    print("🎯 Proceso completado. Nombres homogeneizados sin sufijos de split.")


if __name__ == "__main__":
    main()

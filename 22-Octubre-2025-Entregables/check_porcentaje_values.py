#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_porcentaje_values.py
--------------------------
Script para revisar los valores de la columna 'porcentajeuso' en todos los archivos CSV
y identificar posibles problemas con valores muy grandes.

Uso:
    python check_porcentaje_values.py
    python check_porcentaje_values.py --input salida-csv-zonas-wifi-agrupados-sumados/estandarizados
"""

import os
import argparse
import sys
import pandas as pd
import numpy as np


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Revisa valores de la columna 'porcentajeuso' en archivos CSV."
    )
    parser.add_argument(
        "--input", "-i", 
        default="salida-csv-zonas-wifi-agrupados-sumados/estandarizados",
        help="Carpeta de entrada con archivos CSV estandarizados"
    )
    parser.add_argument(
        "--output", "-o",
        default="reporte_porcentaje_values.txt",
        help="Archivo de salida para el reporte (default: reporte_porcentaje_values.txt)"
    )
    return parser.parse_args()


def list_csvs(folder):
    """Lista todos los archivos CSV en la carpeta."""
    csv_files = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.csv'):
            csv_files.append(os.path.join(folder, filename))
    return sorted(csv_files)


def analyze_porcentaje_column(df, filename):
    """Analiza la columna porcentajeuso de un DataFrame."""
    if 'porcentajeuso' not in df.columns:
        return {
            'file': filename,
            'status': 'NO_COLUMN',
            'message': 'Columna porcentajeuso no encontrada'
        }
    
    col = df['porcentajeuso']
    
    # Estadísticas básicas
    stats = {
        'file': filename,
        'status': 'OK',
        'total_rows': len(df),
        'non_null_count': col.notna().sum(),
        'null_count': col.isna().sum(),
        'unique_values': col.nunique(),
        'min_value': col.min() if col.notna().any() else None,
        'max_value': col.max() if col.notna().any() else None,
        'mean_value': col.mean() if col.notna().any() else None,
        'median_value': col.median() if col.notna().any() else None,
        'std_value': col.std() if col.notna().any() else None,
        'values_in_range_0_100': None,
        'values_above_100': None,
        'values_above_1000': None,
        'values_in_scientific_notation': None,
        'sample_values': None,
        'issues': []
    }
    
    # Verificar si hay valores
    if col.notna().any():
        # Contar valores en diferentes rangos
        numeric_values = pd.to_numeric(col, errors='coerce')
        stats['values_in_range_0_100'] = ((numeric_values >= 0) & (numeric_values <= 100)).sum()
        stats['values_above_100'] = (numeric_values > 100).sum()
        stats['values_above_1000'] = (numeric_values > 1000).sum()
        
        # Detectar valores en notación científica (muy grandes)
        very_large = numeric_values > 1e10
        stats['values_in_scientific_notation'] = very_large.sum()
        
        # Obtener muestra de valores
        sample_size = min(10, len(col.dropna()))
        stats['sample_values'] = col.dropna().head(sample_size).tolist()
        
        # Identificar problemas
        if stats['values_above_100'] > 0:
            stats['issues'].append(f"{stats['values_above_100']} valores > 100")
        if stats['values_above_1000'] > 0:
            stats['issues'].append(f"{stats['values_above_1000']} valores > 1000")
        if stats['values_in_scientific_notation'] > 0:
            stats['issues'].append(f"{stats['values_in_scientific_notation']} valores en notación científica")
        
        if stats['max_value'] and stats['max_value'] > 1e6:
            stats['status'] = 'ISSUES'
            stats['issues'].append(f"Valor máximo muy grande: {stats['max_value']:.2e}")
        elif stats['issues']:
            stats['status'] = 'WARNING'
    
    return stats


def generate_report(all_stats, output_file):
    """Genera un reporte detallado."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE ANÁLISIS DE COLUMNA 'porcentajeuso'\n")
        f.write("=" * 60 + "\n\n")
        
        # Resumen general
        total_files = len(all_stats)
        files_with_issues = len([s for s in all_stats if s['status'] == 'ISSUES'])
        files_with_warnings = len([s for s in all_stats if s['status'] == 'WARNING'])
        files_ok = len([s for s in all_stats if s['status'] == 'OK'])
        files_no_column = len([s for s in all_stats if s['status'] == 'NO_COLUMN'])
        
        f.write(f"RESUMEN GENERAL:\n")
        f.write(f"- Total archivos procesados: {total_files}\n")
        f.write(f"- Archivos con problemas: {files_with_issues}\n")
        f.write(f"- Archivos con advertencias: {files_with_warnings}\n")
        f.write(f"- Archivos OK: {files_ok}\n")
        f.write(f"- Archivos sin columna: {files_no_column}\n\n")
        
        # Archivos con problemas
        if files_with_issues > 0:
            f.write("ARCHIVOS CON PROBLEMAS:\n")
            f.write("-" * 30 + "\n")
            for stat in all_stats:
                if stat['status'] == 'ISSUES':
                    f.write(f"\n📁 {stat['file']}\n")
                    f.write(f"   Problemas: {', '.join(stat['issues'])}\n")
                    f.write(f"   Valor máximo: {stat['max_value']:.2e}\n")
                    f.write(f"   Valor mínimo: {stat['min_value']:.2e}\n")
                    f.write(f"   Muestra: {stat['sample_values'][:5]}\n")
        
        # Archivos con advertencias
        if files_with_warnings > 0:
            f.write("\n\nARCHIVOS CON ADVERTENCIAS:\n")
            f.write("-" * 30 + "\n")
            for stat in all_stats:
                if stat['status'] == 'WARNING':
                    f.write(f"\n⚠️  {stat['file']}\n")
                    f.write(f"   Advertencias: {', '.join(stat['issues'])}\n")
                    f.write(f"   Valor máximo: {stat['max_value']:.2e}\n")
                    f.write(f"   Muestra: {stat['sample_values'][:3]}\n")
        
        # Estadísticas detalladas
        f.write("\n\nESTADÍSTICAS DETALLADAS:\n")
        f.write("-" * 30 + "\n")
        for stat in all_stats:
            if stat['status'] not in ['NO_COLUMN']:
                f.write(f"\n📊 {stat['file']}\n")
                f.write(f"   Filas totales: {stat['total_rows']}\n")
                f.write(f"   Valores no nulos: {stat['non_null_count']}\n")
                f.write(f"   Valores únicos: {stat['unique_values']}\n")
                if stat['min_value'] is not None:
                    f.write(f"   Mínimo: {stat['min_value']:.6f}\n")
                    f.write(f"   Máximo: {stat['max_value']:.6f}\n")
                    f.write(f"   Promedio: {stat['mean_value']:.6f}\n")
                    f.write(f"   Mediana: {stat['median_value']:.6f}\n")
                if stat['values_in_range_0_100'] is not None:
                    f.write(f"   Valores 0-100: {stat['values_in_range_0_100']}\n")
                    f.write(f"   Valores >100: {stat['values_above_100']}\n")
                    f.write(f"   Valores >1000: {stat['values_above_1000']}\n")
                    f.write(f"   Valores notación científica: {stat['values_in_scientific_notation']}\n")


def main():
    """Función principal."""
    args = parse_args()
    
    if not os.path.isdir(args.input):
        print(f"❌ No existe la carpeta: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    csv_files = list_csvs(args.input)
    if not csv_files:
        print("⚠️  No se encontraron archivos CSV en la carpeta.")
        sys.exit(0)
    
    print(f"🔍 Analizando {len(csv_files)} archivos CSV...")
    
    all_stats = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"   📁 Procesando: {filename}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            stats = analyze_porcentaje_column(df, filename)
            all_stats.append(stats)
            
            # Mostrar estado en consola
            if stats['status'] == 'ISSUES':
                print(f"      ❌ PROBLEMAS: {', '.join(stats['issues'])}")
            elif stats['status'] == 'WARNING':
                print(f"      ⚠️  ADVERTENCIAS: {', '.join(stats['issues'])}")
            elif stats['status'] == 'NO_COLUMN':
                print(f"      ℹ️  Sin columna porcentajeuso")
            else:
                print(f"      ✅ OK")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            all_stats.append({
                'file': filename,
                'status': 'ERROR',
                'message': str(e)
            })
    
    # Generar reporte
    print(f"\n📝 Generando reporte: {args.output}")
    generate_report(all_stats, args.output)
    
    # Resumen final
    files_with_issues = len([s for s in all_stats if s['status'] == 'ISSUES'])
    files_with_warnings = len([s for s in all_stats if s['status'] == 'WARNING'])
    
    print(f"\n🎉 Análisis completado!")
    print(f"📊 Total archivos: {len(csv_files)}")
    print(f"❌ Con problemas: {files_with_issues}")
    print(f"⚠️  Con advertencias: {files_with_warnings}")
    print(f"📄 Reporte guardado en: {args.output}")


if __name__ == "__main__":
    main()

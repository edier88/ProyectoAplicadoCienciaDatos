import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================
# CONFIGURACIÓN
# ============================
CARPETA_CSV = "csv-zonas-wifi-separados"

# Columnas categóricas a completar
COLUMNAS_CATEGORICAS = ['AREA', 'NOMBRE_ZONA', 'COMUNA', 'MODEL']

# ============================
# FUNCIONES
# ============================

def completar_columnas_categoricas(df):
    """
    Completa columnas categóricas usando forward fill, backward fill y moda.
    """
    df_work = df.copy()
    
    # Contar valores faltantes antes
    faltantes_antes = {}
    for col in COLUMNAS_CATEGORICAS:
        if col in df_work.columns:
            faltantes_antes[col] = df_work[col].isna().sum()
        else:
            faltantes_antes[col] = 0
    
    # Estrategia 1: Forward fill (rellenar hacia adelante)
    # Si una fila tiene el mismo NOMBRE_ZONA que la anterior, probablemente las demás columnas también son iguales
    for col in COLUMNAS_CATEGORICAS:
        if col in df_work.columns:
            # Forward fill
            df_work[col] = df_work[col].ffill()
    
    # Estrategia 2: Backward fill (rellenar hacia atrás)
    # Para valores al inicio del dataset
    for col in COLUMNAS_CATEGORICAS:
        if col in df_work.columns:
            # Backward fill solo si aún hay NaN
            df_work[col] = df_work[col].bfill()
    
    # Estrategia 3: Si NOMBRE_ZONA está disponible, usar moda por zona
    if 'NOMBRE_ZONA' in df_work.columns:
        # Identificar zonas únicas que tienen datos completos
        zonas_con_datos = df_work[df_work['NOMBRE_ZONA'].notna()]['NOMBRE_ZONA'].unique()
        
        for zona in zonas_con_datos:
            # Filtrar datos de esta zona
            mask_zona = df_work['NOMBRE_ZONA'] == zona
            
            # Para cada columna categórica, usar la moda de esa zona
            for col in COLUMNAS_CATEGORICAS:
                if col in df_work.columns and col != 'NOMBRE_ZONA':
                    # Obtener valores no nulos de esta zona
                    valores_zona = df_work.loc[mask_zona, col]
                    valores_no_nulos = valores_zona.dropna()
                    
                    if len(valores_no_nulos) > 0:
                        # Calcular moda
                        moda = valores_no_nulos.mode()
                        if len(moda) > 0:
                            valor_moda = moda.iloc[0]
                            # Rellenar NaN de esta zona con la moda
                            df_work.loc[mask_zona & df_work[col].isna(), col] = valor_moda
    
    # Estrategia 4: Si aún hay NaN, usar moda global
    for col in COLUMNAS_CATEGORICAS:
        if col in df_work.columns:
            valores_no_nulos = df_work[col].dropna()
            if len(valores_no_nulos) > 0 and df_work[col].isna().any():
                moda_global = valores_no_nulos.mode()
                if len(moda_global) > 0:
                    df_work[col].fillna(moda_global.iloc[0], inplace=True)
    
    # Contar valores faltantes después
    faltantes_despues = {}
    for col in COLUMNAS_CATEGORICAS:
        if col in df_work.columns:
            faltantes_despues[col] = df_work[col].isna().sum()
        else:
            faltantes_despues[col] = 0
    
    return df_work, faltantes_antes, faltantes_despues

def procesar_csv(archivo):
    """
    Procesa un CSV completando las columnas categóricas faltantes.
    """
    try:
        nombre = os.path.basename(archivo)
        print(f"\nProcesando: {nombre}")
        
        # Leer CSV
        df = pd.read_csv(archivo, encoding='utf-8')
        
        # Verificar que existen las columnas categóricas
        columnas_existentes = [col for col in COLUMNAS_CATEGORICAS if col in df.columns]
        if not columnas_existentes:
            print(f"  ⚠️  No se encontraron columnas categóricas en {nombre}")
            return False
        
        # Completar columnas categóricas
        df_completado, faltantes_antes, faltantes_despues = completar_columnas_categoricas(df)
        
        # Guardar archivo
        df_completado.to_csv(archivo, index=False, encoding='utf-8')
        
        # Mostrar estadísticas
        print(f"  ✅ Columnas categóricas completadas:")
        total_antes = sum(faltantes_antes.values())
        total_despues = sum(faltantes_despues.values())
        total_completados = total_antes - total_despues
        
        for col in COLUMNAS_CATEGORICAS:
            if col in df.columns:
                antes = faltantes_antes[col]
                despues = faltantes_despues[col]
                completados = antes - despues
                if antes > 0 or despues > 0:
                    print(f"     {col}: {antes} → {despues} faltantes ({completados} completados)")
        
        if total_completados > 0:
            print(f"     Total: {total_antes} → {total_despues} faltantes ({total_completados} completados)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error procesando {nombre}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================
# PROGRAMA PRINCIPAL
# ============================

def main():
    # Verificar que la carpeta existe
    if not os.path.exists(CARPETA_CSV):
        print(f"❌ Error: La carpeta '{CARPETA_CSV}' no existe.")
        return
    
    # Buscar archivos CSV
    carpeta = Path(CARPETA_CSV)
    archivos_csv = list(carpeta.glob('*.csv'))
    
    if not archivos_csv:
        print(f"❌ No se encontraron archivos CSV en '{CARPETA_CSV}'")
        return
    
    print(f"📁 Se encontraron {len(archivos_csv)} archivos CSV para procesar")
    print(f"\n🔧 Estrategias de completado:")
    print(f"   1. Forward fill (rellenar hacia adelante)")
    print(f"   2. Backward fill (rellenar hacia atrás)")
    print(f"   3. Moda por zona (usar valores más comunes de cada zona)")
    print(f"   4. Moda global (usar valor más común del dataset)")
    
    # Procesar cada archivo
    exitosos = 0
    fallidos = 0
    
    for archivo in archivos_csv:
        if procesar_csv(archivo):
            exitosos += 1
        else:
            fallidos += 1
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"✅ Procesamiento completado:")
    print(f"   Archivos procesados exitosamente: {exitosos}")
    if fallidos > 0:
        print(f"   Archivos con errores: {fallidos}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


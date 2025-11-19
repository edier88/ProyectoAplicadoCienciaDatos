import os
import pandas as pd
from pathlib import Path

# ============================
# CONFIGURACIÓN
# ============================
CARPETA_CSV = "csv-zonas-wifi-separados-PruebaEdier"

# ============================
# FUNCIONES
# ============================

def detectar_columna_fecha(df):
    """
    Detecta automáticamente el nombre de la columna de fecha.
    """
    posibles_nombres = ['FECHA_CONEXION', 'FECHA.CONEXION', 'FECHA CONEXIÓN', 'FECHA CONEXION']
    
    for nombre_posible in posibles_nombres:
        if nombre_posible in df.columns:
            return nombre_posible
    
    return None

def completar_fechas_faltantes(archivo):
    """
    Completa las fechas faltantes en un CSV creando registros para cada día faltante.
    """
    try:
        nombre = os.path.basename(archivo)
        print(f"\nProcesando: {nombre}")
        
        # Leer CSV
        df = pd.read_csv(archivo, encoding='utf-8')
        
        # Detectar nombre de columna de fecha
        columna_fecha = detectar_columna_fecha(df)
        
        if columna_fecha is None:
            print(f"  ❌ No se encontró columna de fecha en {nombre}")
            return False
        
        filas_antes = len(df)
        
        # Convertir fecha a datetime
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], format='%Y-%m-%d', errors='coerce')
        
        # Eliminar filas con fechas inválidas
        df = df.dropna(subset=[columna_fecha])
        
        if len(df) == 0:
            print(f"  ❌ No hay fechas válidas en {nombre}")
            return False
        
        # Establecer fecha como índice
        df = df.set_index(columna_fecha)
        
        # Completar fechas faltantes con frecuencia diaria
        # El día que no encuentre lo creará con las demás columnas en NaN
        df = df.asfreq('D')
        
        # Ordenar por fecha (ascendente)
        df = df.sort_index()
        
        # Convertir columnas de tipo día a Int64 (permite NaN)
        columnas_int64 = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO']
        for col in columnas_int64:
            if col in df.columns:
                df[col] = df[col].astype('Int64')
        
        # Resetear índice: FECHA_CONEXION vuelve a ser una columna normal
        df_reset = df.reset_index()
        
        # Renombrar la columna del índice si tiene un nombre diferente
        if df_reset.columns[0] != 'FECHA_CONEXION':
            df_reset.rename(columns={df_reset.columns[0]: 'FECHA_CONEXION'}, inplace=True)
        
        # Guardar archivo (sobrescribir original)
        df_reset.to_csv(archivo, index=False, encoding='utf-8')
        
        filas_despues = len(df_reset)
        fechas_agregadas = filas_despues - filas_antes
        
        print(f"  ✅ Completado: {filas_antes} → {filas_despues} filas "
              f"({fechas_agregadas} fechas agregadas)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error procesando {nombre}: {e}")
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
    
    # Procesar cada archivo
    exitosos = 0
    fallidos = 0
    
    for archivo in archivos_csv:
        if completar_fechas_faltantes(archivo):
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
        

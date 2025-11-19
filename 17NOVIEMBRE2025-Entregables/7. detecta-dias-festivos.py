import pandas as pd
import holidays
import os
from pathlib import Path

# ============================
# CONFIGURACIÓN
# ============================
CARPETA_CSV = "csv-zonas-wifi-separados-PruebaEdier"  # Carpeta donde están los CSV

# ============================
# FUNCIONES
# ============================

def detectar_tipo_dia(fecha, festivos_colombia):
    """
    Determina el tipo de día basado en la fecha.
    Retorna: (dia_semana, es_laboral, es_fin_semana, es_festivo)
    """
    # Día de la semana (0=Lunes, 6=Domingo)
    dia_semana = fecha.weekday()
    
    # Verificar si es festivo
    nombre_festivo = festivos_colombia.get(fecha.date())
    es_festivo = 1 if nombre_festivo is not None else 0
    
    # Si es festivo, no es laboral ni fin de semana
    if es_festivo:
        return dia_semana, 0, 0, 1
    
    # Si no es festivo, determinar según día de la semana
    if dia_semana < 5:  # Lunes a Viernes (0-4)
        return dia_semana, 1, 0, 0
    else:  # Sábado (5) o Domingo (6)
        return dia_semana, 0, 1, 0

def procesar_csv(archivo, festivos_colombia):
    """
    Procesa un archivo CSV agregando columnas de días festivos y tipo de día.
    """
    try:
        nombre = os.path.basename(archivo)
        print(f"Procesando: {nombre}")
        
        # Leer CSV
        df = pd.read_csv(archivo, encoding='utf-8')
        
        # Detectar nombre de columna de fecha (puede variar)
        columna_fecha = None
        posibles_nombres = ['FECHA CONEXIÓN', 'FECHA.CONEXION', 'FECHA CONEXION', 'FECHA_CONEXION']
        
        for nombre_posible in posibles_nombres:
            if nombre_posible in df.columns:
                columna_fecha = nombre_posible
                break
        
        if columna_fecha is None:
            print(f"  ⚠️  Advertencia: No se encontró columna de fecha en {nombre}")
            return False
        
        # Convertir fecha a datetime
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')
        
        # Eliminar filas con fechas inválidas
        filas_antes = len(df)
        df = df.dropna(subset=[columna_fecha])
        filas_eliminadas = filas_antes - len(df)
        if filas_eliminadas > 0:
            print(f"  ⚠️  Se eliminaron {filas_eliminadas} filas con fechas inválidas")
        
        # Aplicar función para detectar tipo de día
        resultados = df[columna_fecha].apply(
            lambda fecha: detectar_tipo_dia(fecha, festivos_colombia)
        )
        
        # Extraer resultados en columnas separadas
        df['DIA_SEMANA'] = [r[0] for r in resultados]
        df['LABORAL'] = [r[1] for r in resultados]
        df['FIN_DE_SEMANA'] = [r[2] for r in resultados]
        df['FESTIVO'] = [r[3] for r in resultados]
        
        # Convertir a Int64 (permite NaN)
        df['DIA_SEMANA'] = df['DIA_SEMANA'].astype('Int64')
        df['LABORAL'] = df['LABORAL'].astype('Int64')
        df['FIN_DE_SEMANA'] = df['FIN_DE_SEMANA'].astype('Int64')
        df['FESTIVO'] = df['FESTIVO'].astype('Int64')
        
        # Guardar archivo (sobrescribir original)
        df.to_csv(archivo, index=False, encoding='utf-8')
        
        # Estadísticas
        total_festivos = df['FESTIVO'].sum()
        total_laborales = df['LABORAL'].sum()
        total_fin_semana = df['FIN_DE_SEMANA'].sum()
        
        print(f"  ✅ Procesado: {len(df)} filas | "
              f"Festivos: {total_festivos} | "
              f"Laborales: {total_laborales} | "
              f"Fin de semana: {total_fin_semana}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error al procesar {nombre}: {e}")
        return False

# ============================
# PROGRAMA PRINCIPAL
# ============================

def main():
    # Verificar que la carpeta existe
    if not os.path.exists(CARPETA_CSV):
        print(f"❌ Error: La carpeta '{CARPETA_CSV}' no existe.")
        return
    
    # Inicializar festivos de Colombia
    festivos_colombia = holidays.Colombia()
    
    # Buscar archivos CSV
    carpeta = Path(CARPETA_CSV)
    archivos_csv = list(carpeta.glob('*.csv'))
    
    if not archivos_csv:
        print(f"❌ No se encontraron archivos CSV en '{CARPETA_CSV}'")
        return
    
    print(f"📁 Se encontraron {len(archivos_csv)} archivos CSV para procesar\n")
    
    # Procesar cada archivo
    exitosos = 0
    fallidos = 0
    
    for archivo in archivos_csv:
        if procesar_csv(archivo, festivos_colombia):
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

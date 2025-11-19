import os
import pandas as pd
import glob

# 21-09-2025
# lee la carpeta csv-zonas-wifi-1AP-todas-las-columnas

def procesar_csv_en_carpeta(carpeta):
    """
    Lee todos los archivos CSV en una carpeta y realiza:
    1. One Hot Encoding para la columna 'tipo_dia'
    2. Conversión de valores cualitativos a numéricos para 'dia_semana'
    """
    patron = os.path.join(carpeta, "*.csv")
    archivos_csv = glob.glob(patron)
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en la carpeta: {carpeta}")
        return
    
    print(f"Se encontraron {len(archivos_csv)} archivos CSV para procesar")
    
    # Mapeo para la conversión de dia_semana
    mapeo_dia_semana = {
        'Lunes': 0,
        'Martes': 1,
        'Miércoles': 2,
        'Jueves': 3,
        'Viernes': 4,
        'Sábado': 5,
        'Domingo': 6
    }
    
    for archivo in archivos_csv:
        try:
            print(f"\nProcesando: {os.path.basename(archivo)}")
            df = pd.read_csv(archivo)
            
            print(f"Filas originales: {len(df)}")
            print(f"Columnas originales: {list(df.columns)}")
            
            # 1. One Hot Encoding para 'TIPO_DIA'
            if 'TIPO_DIA' in df.columns:
                # Crear las tres columnas binarias
                df['LABORAL'] = (df['TIPO_DIA'] == 'Laboral').astype(int)
                df['FIN_DE_SEMANA'] = (df['TIPO_DIA'] == 'Fin de Semana').astype(int)
                df['FESTIVO'] = (df['TIPO_DIA'] == 'Festivo').astype(int)
                
                # Eliminar la columna original
                df = df.drop(columns=['TIPO_DIA'])
                print("✓ One Hot Encoding aplicado a 'TIPO_DIA'")
                print("  - Columnas creadas: LABORAL, FIN_DE_SEMANA, FESTIVO")
            else:
                print("ℹ Columna 'TIPO_DIA' no encontrada")
            
            # 2. Conversión de 'DIA_SEMANA' a valores numéricos
            if 'DIA_SEMANA' in df.columns:
                # Aplicar el mapeo
                df['DIA_SEMANA'] = df['DIA_SEMANA'].map(mapeo_dia_semana)
                
                # Verificar si hay valores no mapeados
                if df['DIA_SEMANA'].isnull().any():
                    valores_unicos = df['DIA_SEMANA'].dropna().unique()
                    print(f"⚠ Advertencia: Algunos valores no pudieron ser mapeados")
                    print(f"  Valores únicos encontrados: {valores_unicos}")
                
                print("✓ Columna 'DIA_SEMANA' convertida a valores numéricos")
                print(f"  - Mapeo aplicado: {mapeo_dia_semana}")
            else:
                print("ℹ Columna 'DIA_SEMANA' no encontrada")
            
            # Guardar el archivo procesado
            df.to_csv(archivo, index=False, encoding='utf-8')
            print(f"✓ Archivo guardado: {os.path.basename(archivo)}")
            print(f"✓ Filas procesadas: {len(df)}")
            print(f"✓ Columnas finales: {list(df.columns)}")
                
        except Exception as e:
            print(f"✗ Error procesando {archivo}: {str(e)}")
    
    print("\n¡Procesamiento completado!")

if __name__ == "__main__":
    # Carpeta relativa al script
    carpeta = os.path.join(os.path.dirname(__file__), "csv-zonas-wifi-1AP-todas-las-columnas")
    
    if os.path.exists(carpeta):
        procesar_csv_en_carpeta(carpeta)
    else:
        print(f"La carpeta '{carpeta}' no existe. Verifica que esté en el mismo directorio del script.")
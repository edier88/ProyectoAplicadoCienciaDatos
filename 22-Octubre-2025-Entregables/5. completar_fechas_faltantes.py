import os
import pandas as pd
from pathlib import Path

target_folder = Path("csv-zonas-wifi-1AP-todas-las-columnas")
#target_folder = Path("csv-zonas-wifi-1AP-todas-las-columnas_2")
#target_folder.mkdir(exist_ok=True)
read_folder = "csv-zonas-wifi-1AP-todas-las-columnas/"

if os.path.exists(read_folder):
    print(f"carpeta existe '{read_folder}'")
else:
    print(f"La carpeta '{read_folder}' no existe. Verifica que esté en el mismo directorio del script.")

archivos_csv = [f for f in os.listdir(read_folder) if f.endswith(".csv")]

for archivo in archivos_csv:
    
    try:
        print(f"\nProcesando: {os.path.basename(archivo)}")
        path_csv = os.path.join(read_folder, archivo)
        df = pd.read_csv(path_csv)

        # Preparación de la fecha
        # # ==============================================================================
        df['FECHA.CONEXION'] = pd.to_datetime(df['FECHA.CONEXION'], format='%Y-%m-%d')
        df = df.set_index('FECHA.CONEXION')
        df = df.asfreq('D') # Frecuencia Diario, el día que no encuentre lo creará  con las demás columnas en NaN
        
        # En caso de que esten las fechas desorganizadas se organizan de forma ascendent
        df = df.sort_index()
        
        # Int64 con la "I" mayúscula indica que ignore los NaN
        df['DIA_SEMANA'] = df['DIA_SEMANA'].astype('Int64')
        df['LABORAL'] = df['LABORAL'].astype('Int64')
        df['FIN_DE_SEMANA'] = df['FIN_DE_SEMANA'].astype('Int64')
        df['FESTIVO'] = df['FESTIVO'].astype('Int64')
        
        # Se resetea FECHA.CONEXION como index. Vuelve a ser una columna normal
        df_reset = df.reset_index()

        nombre_archivo = path_csv.split('/')
        df_reset.to_csv(target_folder / nombre_archivo[1], index=False, encoding='utf-8')
        print(f"✅ Guardado en {target_folder}")
    
    except Exception as e:
        print(f"   ❌ Error procesando: {e}")
        

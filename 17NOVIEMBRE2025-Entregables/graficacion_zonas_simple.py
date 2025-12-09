import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from skforecast.plot import set_dark_theme

import glob
import os
from pathlib import Path

GRAF_DIR = "Graficas_Zonas_Sin_Imputacion"
os.makedirs(GRAF_DIR, exist_ok=True)

ORIGEN = "csv-zonas-wifi-separados-PruebaEdier - copia/"
#DESTINO_METRICAS = "Random_Forest_Metricas"
DESTINO_METRICAS = Path("Graficas_Zonas_Sin_Imputacion")
os.makedirs(DESTINO_METRICAS, exist_ok=True)

#carpeta = os.path.join(os.path.dirname(__file__), "csv-zonas-wifi-1AP-todas-las-columnas")
#print(carpeta)

archivos = glob.glob(os.path.join(ORIGEN, "*.csv"))

for archivo in archivos:
    nombre_zona = os.path.basename(archivo)
    print(f"\nProcesando: {nombre_zona}")
    
    #df = pd.read_csv(os.path.join(carpeta,"001_ZW Parque Ingenio-test2.csv"))
    df = pd.read_csv(archivo)
    
    # ==============================================================================
    # Preparación del dataset
    # ==============================================================================

    df['FECHA_CONEXION'] = pd.to_datetime(df['FECHA_CONEXION'], format='%Y-%m-%d')
    df = df.set_index('FECHA_CONEXION')
    df = df.asfreq('D') # Fechas con frecuencia diario. En caso de que falte algún día se crea y todas las demás variables se ponen NaN
    
    df = df.sort_index() # En caso de que esten las fechas desorganizadas se organizan de forma ascendente
    
    # Hacer método para detectar si hay datos faltantes y no proseguir con el modelado
    df.isnull().any(axis=1).mean()
    
    rows_with_na = df[df.isna().any(axis=1)]
    print("\nRows with any NA values:")
    print(rows_with_na)
    
    rows, columns = df.shape
    print(rows)
    
    # Conversion de los números a los tipos adecuados
    df['DIA_SEMANA'] = df['DIA_SEMANA'].astype('Int64')
    df['LABORAL'] = df['LABORAL'].astype('Int64')
    df['FIN_DE_SEMANA'] = df['FIN_DE_SEMANA'].astype('Int64')
    df['FESTIVO'] = df['FESTIVO'].astype('Int64')
    df['PORCENTAJE_USO'] = df['PORCENTAJE_USO'].astype('Float64')
    df['NUMERO_CONEXIONES'] = df['NUMERO_CONEXIONES'].astype('Float64')
    df['USAGE_KB'] = df['USAGE_KB'].astype('Float64')

    # Graficación serie temporal
    set_dark_theme()
    #plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(25, 4))
    plt.plot(df['USAGE_KB'], label="Train", linewidth=2)
    plt.title(f"{nombre_zona}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, f"{nombre_zona}_serie.png"), dpi=300)
    plt.close()

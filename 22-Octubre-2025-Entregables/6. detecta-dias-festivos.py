import pandas as pd
import holidays
import glob
import os

# Carpeta donde están CSV
#ORIGEN = "csv-zonas-wifi-1AP-todas-las-columnas_2/"  # Carpeta donde están los CSV originales
#DESTINO = "csv-zonas-wifi-festivos/" # Carpeta donde se guardarán los CSV con los festivos
ORIGEN = "csv-zonas-wifi-1AP-todas-las-columnas/"  # Carpeta donde están los CSV originales
DESTINO = "csv-zonas-wifi-1AP-todas-las-columnas/" # Carpeta donde se guardarán los CSV con los festivos

os.makedirs(DESTINO, exist_ok=True)

# Festivos de Colombia
festivos = holidays.Colombia()
#print("festivos:")
#print(map(festivos.get))

# Buscar archivos CSV
archivos = glob.glob(os.path.join(ORIGEN, "*.csv"))

for archivo in archivos:
    nombre = os.path.basename(archivo)
    print(f"Procesando: {nombre}")

    # Leer CSV
    df = pd.read_csv(archivo)

    # Convertir fecha
    df["FECHA.CONEXION"] = pd.to_datetime(df["FECHA.CONEXION"])

    # Día semana
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Lunes", 'DIA_SEMANA'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Lunes", 'LABORAL'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Lunes", 'FIN_DE_SEMANA'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Lunes", 'FESTIVO'] = 0

    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Martes", 'DIA_SEMANA'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Martes", 'LABORAL'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Martes", 'FIN_DE_SEMANA'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Martes", 'FESTIVO'] = 0

    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Miércoles", 'DIA_SEMANA'] = 2
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Miércoles", 'LABORAL'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Miércoles", 'FIN_DE_SEMANA'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Miércoles", 'FESTIVO'] = 0

    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Jueves", 'DIA_SEMANA'] = 3
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Jueves", 'LABORAL'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Jueves", 'FIN_DE_SEMANA'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Jueves", 'FESTIVO'] = 0

    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Viernes", 'DIA_SEMANA'] = 4
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Viernes", 'LABORAL'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Viernes", 'FIN_DE_SEMANA'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Viernes", 'FESTIVO'] = 0

    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Sábado", 'DIA_SEMANA'] = 5
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Sábado", 'LABORAL'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Sábado", 'FIN_DE_SEMANA'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Sábado", 'FESTIVO'] = 0

    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Domingo", 'DIA_SEMANA'] = 6
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Domingo", 'LABORAL'] = 0
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Domingo", 'FIN_DE_SEMANA'] = 1
    df.loc[ df["FECHA.CONEXION"].dt.day_name(locale='es_CO') == "Domingo", 'FESTIVO'] = 0

    #print(df["FECHA.CONEXION"].isin(festivos))

    # Nombre del festivo
    df["NOMBRE_FESTIVO"] = df["FECHA.CONEXION"].map(festivos.get)

    # Festivo
    df.loc[ df["NOMBRE_FESTIVO"].notna() == True, 'LABORAL'] = 0
    df.loc[ df["NOMBRE_FESTIVO"].notna() == True, 'FIN_DE_SEMANA'] = 0
    df.loc[ df["NOMBRE_FESTIVO"].notna() == True, 'FESTIVO'] = 1

    # Int64 con la "I" mayúscula indica que ignore los NaN
    df['DIA_SEMANA'] = df['DIA_SEMANA'].astype('Int64')
    df['LABORAL'] = df['LABORAL'].astype('Int64')
    df['FIN_DE_SEMANA'] = df['FIN_DE_SEMANA'].astype('Int64')
    df['FESTIVO'] = df['FESTIVO'].astype('Int64')

    df_new = df.drop("NOMBRE_FESTIVO", axis=1)
    
  

    # Día semana
    #df["DIA_SEMANA"] = df["FECHA.CONEXION"].dt.day_name(locale='es_CO')

    # Festivo
    #df["ES_FESTIVO"] = df["FECHA.CONEXION"].isin(festivos)

    

    # Guardar archivo
    df_new.to_csv(os.path.join(DESTINO, nombre), index=False)

print("✔️ Procesamiento completado para todos los CSV.")

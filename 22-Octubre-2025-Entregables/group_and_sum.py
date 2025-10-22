import pandas as pd
import os

# Cargar el CSV (asegúrate de que el separador decimal sea correcto)
df = pd.read_csv("Zonas-con-dos-AP/001_ZW Parque Ingenio-test.csv", decimal=",")


# Columnas por las cuales agrupar (ajusta según necesites)
columnas_agrupacion = [
    "FECHA.CONEXION", 
    "AREA", 
    "NOMBRE.ZONA", 
    "COMUNA", 
    "MODEL",
    "es_festivo",
    "tipo_dia",
    "dia_semana",
    "LATITUD",
    "LONGITUD"
]

# Agrupar y sumar las columnas numéricas (USAGE.KB y PORCENTAJE.USO)
df_agrupado = df.groupby(columnas_agrupacion, as_index=False).agg({
    "NUMERO.CONEXIONES": "sum",
    "USAGE.KB": "sum",
    "PORCENTAJE.USO": "sum"
})

# Guardar el resultado en un nuevo CSV
df_agrupado.to_csv("datos_agrupados.csv", index=False, decimal=",")

print("¡Archivo guardado como 'datos_agrupados.csv'!")
print(f"Filas originales: {len(df)} → Filas agrupadas: {len(df_agrupado)}")

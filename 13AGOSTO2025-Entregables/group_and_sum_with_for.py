import pandas as pd
import os
from pathlib import Path

# Configuración inicial
carpeta_entrada = Path("test")  # Cambia esto a tu ruta real
carpeta_salida = carpeta_entrada / "resultados_agrupados"
carpeta_salida.mkdir(exist_ok=True)  # Crear carpeta de salida si no existe

# Columnas para agrupación (ajusta según necesites)
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

# Procesar cada archivo CSV en la carpeta
for archivo in carpeta_entrada.glob("*.csv"):
    # Cargar el CSV
    df = pd.read_csv(archivo, decimal=",")
    
    # Agrupar y sumar
    df_agrupado = df.groupby(columnas_agrupacion, as_index=False).agg({
        "NUMERO.CONEXIONES": "sum",
        "USAGE.KB": "sum",
        "PORCENTAJE.USO": "sum"  # Cambiar a "mean" si prefieres promedio
    })
    
    # Guardar resultado
    nombre_salida = f"agrupado_{archivo.name}"
    df_agrupado.to_csv(carpeta_salida / nombre_salida, index=False, decimal=",")
    print(f"✅ Procesado: {archivo.name} -> {nombre_salida}")

print("\n🎉 ¡Procesamiento completado!")
print(f"📂 Archivos guardados en: {carpeta_salida}")
import os
import pandas as pd

# ============================
# CONFIGURACIÓN
# ============================

INPUT_CSV = "conexiones_zonas_wifi_septiembre2025.csv"
OUTPUT_FOLDER = "csv-zonas-wifi-separados-PruebaEdier/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\n🚀 INICIO SEPARACIÓN SIMPLE POR ZONA WIFI\n")

# ============================
# CARGAR CSV ORIGINAL (DATIC)
# ============================

# El archivo está separado por ;
df = pd.read_csv(INPUT_CSV, encoding="latin1", sep=";")

print("Columnas detectadas:")
print(df.columns.tolist())

# ============================
# LISTAR TODAS LAS ZONAS
# ============================

if "NOMBRE ZONA" in df.columns:
    zona_col = "NOMBRE ZONA"
elif "ZONA" in df.columns:
    zona_col = "ZONA"
else:
    raise Exception("❌ No se encontró la columna de zona en el CSV.")

zonas = df[zona_col].unique()

print("\nZonas encontradas:")
for z in zonas:
    print(" -", z)

# ============================
# SEPARAR POR ZONA
# ============================

for zona in zonas:
    print(f"\n📌 Exportando datos de: {zona}")

    df_zona = df[df[zona_col] == zona].copy()

    # Nombre seguro del archivo
    nombre_archivo = zona.replace("/", "-").replace("\\", "-") + ".csv"
    salida = os.path.join(OUTPUT_FOLDER, nombre_archivo)

    # Guardar sin modificar nada
    df_zona.to_csv(salida, index=False, encoding="utf-8")

    print(f"✔ Archivo creado: {salida}")

print("\n🎉 SEPARACIÓN COMPLETADA EXITOSAMENTE.\n")

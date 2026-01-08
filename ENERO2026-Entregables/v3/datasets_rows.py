import pandas as pd
import glob
import os
from pathlib import Path

# Archivo que muestra el tamaño en filas de cada dataset y guarda la información en un CSV "datasets_rows"

ORIGEN = "csv-zonas-wifi-separados-man-renumerados/"

DATASETS = Path("datasets_rows")
os.makedirs(DATASETS, exist_ok=True)

archivos = glob.glob(os.path.join(ORIGEN, "*.csv"))

df_lengths = pd.DataFrame(columns=["Zona", "Rows"])

for archivo in archivos:
    
    nombre_zona = os.path.basename(archivo)
    nombre_zona = nombre_zona[:-4]
    nombre_zona = nombre_zona[4:]

    df = pd.read_csv(archivo)
    print(f"{nombre_zona}:")
    length_rows = len(df)
    print(length_rows)

    new_row = pd.DataFrame([{"Zona": nombre_zona, "Rows": length_rows}])
    df_lengths = pd.concat([df_lengths, new_row], ignore_index=True)

df_lengths.to_csv(DATASETS / f"datasets_rows.csv", index=False, encoding='utf-8')

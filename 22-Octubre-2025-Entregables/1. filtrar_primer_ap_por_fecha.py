import os
import pandas as pd
from pathlib import Path

# 15-NOV-2025


def limpiar_csv(path_csv, target_folder):
    """
    Carga un CSV de zona WiFi que tiene dos APs por fecha (duplicados de fecha)
    y conserva únicamente el primer registro de cada día.
    """
    try:
        df = pd.read_csv(path_csv)

        # Normalizar nombre de columna fecha
        if "FECHA.CONEXION" not in df.columns:
            print(f"[ERROR] {path_csv} no tiene columna FECHA.CONEXION")
            return

        # Convertir fecha a datetime
        df["FECHA.CONEXION"] = pd.to_datetime(df["FECHA.CONEXION"], errors="coerce")

        # Eliminar filas sin fecha válida
        df = df.dropna(subset=["FECHA.CONEXION"])

        # Ordenar por fecha para garantizar que tomamos el primer AP del día
        df = df.sort_values("FECHA.CONEXION")

        # Quedarnos SOLO con el primer registro por cada fecha (AP1)
        df_filtrado = df.groupby("FECHA.CONEXION").head(1).reset_index(drop=True)

        # Guardar archivos en la carpeta TARGET_FOLDER
        nombre_archivo = path_csv.split('/')
        print(nombre_archivo[1])
        df_filtrado.to_csv(target_folder / nombre_archivo[1], index=False)

        print(f"[OK] Limpiado: {os.path.basename(path_csv)} → {len(df_filtrado)} filas finales")

    except Exception as e:
        print(f"[ERROR] procesando {path_csv}: {str(e)}")


def procesar_carpeta(read_folder, target_folder):
    """
    Recorre todos los CSV del folder solicitado y aplica la limpieza.
    """
    if not os.path.exists(read_folder):
        print(f"[ERROR] Carpeta no encontrada: {read_folder}")
        return

    archivos = [f for f in os.listdir(read_folder) if f.endswith(".csv")]

    if not archivos:
        print("[INFO] No hay archivos CSV para procesar.")
        return

    print(f"[INFO] Procesando {len(archivos)} archivos...\n")

    for archivo in archivos:
        limpiar_csv(os.path.join(read_folder, archivo), target_folder)

    print("\n[FIN] Limpieza completada.")


# =========================
# CONFIGURACIÓN PRINCIPAL
# =========================

if __name__ == "__main__":
    READ_FOLDER = "csv-zonas-wifi-sin-agrupar-no-sumarizado/"
    TARGET_FOLDER = Path("csv-zonas-wifi-1AP-todas-las-columnas")
    TARGET_FOLDER.mkdir(exist_ok=True)
    procesar_carpeta(READ_FOLDER, TARGET_FOLDER)

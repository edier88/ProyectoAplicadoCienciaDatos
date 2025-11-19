import os
import pandas as pd

# 15-NOV-2025


def limpiar_csv(path_csv):
    """
    Carga un CSV de zona WiFi que tiene dos APs por fecha (duplicados de fecha)
    y conserva únicamente el primer registro de cada día.
    """
    try:
        df = pd.read_csv(path_csv, encoding='utf-8')

        # Detectar nombre de columna de fecha (puede variar según normalización)
        columna_fecha = None
        posibles_nombres = ['FECHA_CONEXION', 'FECHA.CONEXION', 'FECHA CONEXIÓN', 'FECHA CONEXION']
        
        for nombre_posible in posibles_nombres:
            if nombre_posible in df.columns:
                columna_fecha = nombre_posible
                break
        
        if columna_fecha is None:
            print(f"[ERROR] {os.path.basename(path_csv)} no tiene columna de fecha")
            return

        # Guardar número original de filas
        filas_originales = len(df)

        # Convertir fecha a datetime
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")

        # Eliminar filas sin fecha válida
        df = df.dropna(subset=[columna_fecha])
        filas_despues_validacion = len(df)
        filas_eliminadas_invalidas = filas_originales - filas_despues_validacion
        
        if filas_eliminadas_invalidas > 0:
            print(f"  ⚠️  Se eliminaron {filas_eliminadas_invalidas} filas con fechas inválidas")

        # Ordenar por fecha para garantizar que tomamos el primer AP del día
        df = df.sort_values(columna_fecha)

        # Quedarnos SOLO con el primer registro por cada fecha (AP1)
        df_filtrado = df.groupby(columna_fecha).head(1).reset_index(drop=True)

        # Guardar reemplazando archivo
        df_filtrado.to_csv(path_csv, index=False, encoding='utf-8')

        filas_finales = len(df_filtrado)
        filas_eliminadas_duplicados = filas_despues_validacion - filas_finales
        
        print(f"[OK] {os.path.basename(path_csv)}: {filas_originales} → {filas_finales} filas "
              f"({filas_eliminadas_duplicados} duplicados eliminados)")

    except Exception as e:
        print(f"[ERROR] procesando {path_csv}: {str(e)}")


def procesar_carpeta(folder):
    """
    Recorre todos los CSV del folder solicitado y aplica la limpieza.
    """
    from pathlib import Path
    
    if not os.path.exists(folder):
        print(f"[ERROR] Carpeta no encontrada: {folder}")
        return

    # Buscar archivos CSV usando pathlib
    carpeta = Path(folder)
    archivos_csv = list(carpeta.glob('*.csv'))

    if not archivos_csv:
        print("[INFO] No hay archivos CSV para procesar.")
        return

    print(f"[INFO] Procesando {len(archivos_csv)} archivos...\n")

    exitosos = 0
    fallidos = 0
    
    for archivo in archivos_csv:
        try:
            limpiar_csv(archivo)
            exitosos += 1
        except Exception as e:
            print(f"[ERROR] Error procesando {archivo.name}: {e}")
            fallidos += 1

    print(f"\n[FIN] Limpieza completada: {exitosos} exitosos, {fallidos} con errores")


# =========================
# CONFIGURACIÓN PRINCIPAL
# =========================

if __name__ == "__main__":
    FOLDER = "csv-zonas-wifi-separados-PruebaEdier/"
    procesar_carpeta(FOLDER)

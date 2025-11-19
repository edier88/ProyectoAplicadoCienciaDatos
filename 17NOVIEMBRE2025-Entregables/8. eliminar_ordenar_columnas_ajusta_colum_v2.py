# eliminar_ordenar_columnas_ajusta_columnas.py
# 24-09-2025
# 11-10-2025 - se reviso este script y si esta operando normalmente al ejecutar este script se actualiza los csv-zonas-wifi-1AP-todas-las-columnas/ 
# con las columnas eliminadas y ordenadas correctamente y lee la carpeta csv-zonas-wifi-1AP-todas-las-columnas/. 
import os
import pandas as pd
import glob

def procesar_csv_en_carpeta(carpeta):
    """
    Lee todos los archivos CSV en una carpeta, elimina columnas no deseadas,
    convierte PORCENTAJE_USO a numérico y reordena columnas clave al final.
    """
    patron = os.path.join(carpeta, "*.csv")
    archivos_csv = glob.glob(patron)
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en la carpeta: {carpeta}")
        return
    
    print(f"Se encontraron {len(archivos_csv)} archivos CSV para procesar")
    
    # Columnas que se eliminarán
    columnas_a_eliminar = [
        'AREA', 
        'NOMBRE_ZONA', 
        'COMUNA', 
        'MODEL',
        'ES_FESTIVO',
        'LATITUD', 
        'LONGITUD'
    ]
    
    for archivo in archivos_csv:
        try:
            print(f"\nProcesando: {os.path.basename(archivo)}")
            df = pd.read_csv(archivo)
            
            # Guardar el estado original de columnas
            columnas_originales = list(df.columns)
            
            # Eliminar columnas no deseadas
            columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
            if columnas_existentes:
                df = df.drop(columns=columnas_existentes, errors='ignore')
                print(f"✓ Columnas eliminadas: {columnas_existentes}")
            else:
                print("✓ No se encontraron columnas para eliminar en este archivo")
            
            # --- Conversión de PORCENTAJE_USO a numérico ---
            if 'PORCENTAJE_USO' in df.columns:
                df['PORCENTAJE_USO'] = (
                    df['PORCENTAJE_USO']
                    .astype(str)                        # asegurar string
                    .str.replace('%', '', regex=False)  # quitar símbolo %
                    .str.replace(',', '.', regex=False) # cambiar coma por punto
                    .str.strip()
                )
                df['PORCENTAJE_USO'] = pd.to_numeric(df['PORCENTAJE_USO'], errors='coerce')
                print("✓ Columna 'PORCENTAJE_USO' convertida a numérico (coma → punto)")
            else:
                print("ℹ Columna 'PORCENTAJE_USO' no encontrada en este archivo")
            
            # Reordenar columnas: PORCENTAJE_USO al final
            if 'PORCENTAJE_USO' in df.columns:
                otras_columnas = [col for col in df.columns if col != 'PORCENTAJE_USO']
                df = df[otras_columnas + ['PORCENTAJE_USO']]
                print("✓ Columna 'PORCENTAJE_USO' movida al último lugar")
            
            # Reordenar columnas: NUMERO_CONEXIONES al final
            if 'NUMERO_CONEXIONES' in df.columns:
                otras_columnas = [col for col in df.columns if col != 'NUMERO_CONEXIONES']
                df = df[otras_columnas + ['NUMERO_CONEXIONES']]
                print("✓ Columna 'NUMERO_CONEXIONES' movida al último lugar")
            
            # Reordenar columnas: USAGE_KB al final
            if 'USAGE_KB' in df.columns:
                otras_columnas = [col for col in df.columns if col != 'USAGE_KB']
                df = df[otras_columnas + ['USAGE_KB']]
                print("✓ Columna 'USAGE_KB' movida al último lugar")
            
            # Guardar el archivo procesado (sobrescribe original)
            df.to_csv(archivo, index=False, encoding='utf-8')
            print(f"✓ Archivo guardado: {os.path.basename(archivo)}")
                
        except Exception as e:
            print(f"✗ Error procesando {archivo}: {str(e)}")
    
    print("\nProcesamiento completado!")

if __name__ == "__main__":
    # Carpeta relativa al script
    carpeta = os.path.join(os.path.dirname(__file__), "csv-zonas-wifi-de-pamartin")
    
    if os.path.exists(carpeta):
        procesar_csv_en_carpeta(carpeta)
    else:
        print(f"La carpeta '{carpeta}' no existe. Verifica que esté en el mismo directorio del script.")

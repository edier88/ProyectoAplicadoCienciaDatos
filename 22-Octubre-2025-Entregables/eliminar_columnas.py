import os
import pandas as pd
import glob

def procesar_csv_en_carpeta(carpeta):
    """
    Lee todos los archivos CSV en una carpeta y elimina las columnas no deseadas
    """
    patron = os.path.join(carpeta, "*.csv")
    archivos_csv = glob.glob(patron)
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en la carpeta: {carpeta}")
        return
    
    print(f"Se encontraron {len(archivos_csv)} archivos CSV para procesar")
    
    columnas_a_eliminar = [
        'FECHA.CONEXION', 
        'AREA', 
        'NOMBRE.ZONA', 
        'COMUNA', 
        'MODEL',
        'ES_FESTIVO',
        'LATITUD', 
        'LONGITUD'
    ]
    
    for archivo in archivos_csv:
        try:
            print(f"Procesando: {os.path.basename(archivo)}")
            df = pd.read_csv(archivo)
            columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
            
            if columnas_existentes:
                df = df.drop(columns=columnas_existentes, errors='ignore')
                df.to_csv(archivo, index=False, encoding='utf-8')
                print(f"✓ Columnas eliminadas: {columnas_existentes}")
            else:
                print("✓ No se encontraron las columnas especificadas en este archivo")
                
        except Exception as e:
            print(f"✗ Error procesando {archivo}: {str(e)}")
    
    print("Procesamiento completado!")

if __name__ == "__main__":
    # Carpeta relativa al script
    carpeta = os.path.join(os.path.dirname(__file__), "csv-zonas-wifi-agrupados-sumados")
    
    if os.path.exists(carpeta):
        procesar_csv_en_carpeta(carpeta)
    else:
        print(f"La carpeta '{carpeta}' no existe. Verifica que esté en el mismo directorio del script.")

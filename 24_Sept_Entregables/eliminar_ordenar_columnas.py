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
    
    # Eliminando FECHA.CONEXION
    """
    columnas_a_eliminar = [
        'FECHA.CONEXION', 
        'AREA', 
        'NOMBRE.ZONA', 
        'COMUNA', 
        'MODEL',
        'es_festivo',
        'LATITUD', 
        'LONGITUD'
    ]
    """
    # Sin eliminar FECHA.CONEXION
    
    columnas_a_eliminar = [
        'AREA', 
        'NOMBRE.ZONA', 
        'COMUNA', 
        'MODEL',
        'es_festivo',
        'LATITUD', 
        'LONGITUD'
    ]
    

    for archivo in archivos_csv:
        try:
            print(f"Procesando: {os.path.basename(archivo)}")
            df = pd.read_csv(archivo)
            
            # Guardar el estado original de columnas
            columnas_originales = list(df.columns)
            
            # Eliminar columnas no deseadas
            columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
            
            if columnas_existentes:
                df = df.drop(columns=columnas_existentes, errors='ignore')
                print(f"✓ Columnas eliminadas: {columnas_existentes}")
            else:
                print("✓ No se encontraron las columnas especificadas en este archivo")
            
            # Mover la columna "PORCENTAJE.USO" al último lugar si existe
            if 'PORCENTAJE.USO' in df.columns:
                # Obtener todas las columnas excepto "USAGE.KB"
                otras_columnas = [col for col in df.columns if col != 'PORCENTAJE.USO']
                # Reordenar columnas: todas las demás primero, luego "USAGE.KB"
                df = df[otras_columnas + ['PORCENTAJE.USO']]
                print("✓ Columna 'PORCENTAJE.USO' movida al último lugar")
            else:
                print("ℹ Columna 'PORCENTAJE.USO' no encontrada en este archivo")
            
            # Mover la columna "NUMERO.CONEXIONES" al último lugar si existe
            if 'NUMERO.CONEXIONES' in df.columns:
                # Obtener todas las columnas excepto "USAGE.KB"
                otras_columnas = [col for col in df.columns if col != 'NUMERO.CONEXIONES']
                # Reordenar columnas: todas las demás primero, luego "USAGE.KB"
                df = df[otras_columnas + ['NUMERO.CONEXIONES']]
                print("✓ Columna 'NUMERO.CONEXIONES' movida al último lugar")
            else:
                print("ℹ Columna 'NUMERO.CONEXIONES' no encontrada en este archivo")
            
            
            # Mover la columna "USAGE.KB" al último lugar si existe
            if 'USAGE.KB' in df.columns:
                # Obtener todas las columnas excepto "USAGE.KB"
                otras_columnas = [col for col in df.columns if col != 'USAGE.KB']
                # Reordenar columnas: todas las demás primero, luego "USAGE.KB"
                df = df[otras_columnas + ['USAGE.KB']]
                print("✓ Columna 'USAGE.KB' movida al último lugar")
            else:
                print("ℹ Columna 'USAGE.KB' no encontrada en este archivo")
            
            # Guardar el archivo procesado
            df.to_csv(archivo, index=False, encoding='utf-8')
            print(f"✓ Archivo guardado: {os.path.basename(archivo)}")
                
        except Exception as e:
            print(f"✗ Error procesando {archivo}: {str(e)}")
    
    print("Procesamiento completado!")

if __name__ == "__main__":
    # Carpeta relativa al script
    carpeta = os.path.join(os.path.dirname(__file__), "resultados_agrupados")
    
    if os.path.exists(carpeta):
        procesar_csv_en_carpeta(carpeta)
    else:
        print(f"La carpeta '{carpeta}' no existe. Verifica que esté en el mismo directorio del script.")
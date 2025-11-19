import pandas as pd
import os
from pathlib import Path

def convertir_fecha_espanol_a_iso(fecha_str):
    """
    Convierte fecha de formato español (ej: '1-ene-24') a formato ISO (ej: '2024-01-01')
    """
    # Diccionario de meses en español
    meses = {
        'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
    }
    
    try:
        # Dividir la fecha: '1-ene-24' -> ['1', 'ene', '24']
        partes = fecha_str.split('-')
        if len(partes) != 3:
            return fecha_str  # Si no tiene el formato esperado, retornar original
        
        dia = partes[0].zfill(2)  # Asegurar 2 dígitos: '1' -> '01'
        mes_abrev = partes[1].lower()  # Convertir a minúsculas
        anio = partes[2]
        
        # Convertir año de 2 dígitos a 4 dígitos
        # Asumimos que años 00-50 son 2000-2050, y 51-99 son 1951-1999
        anio_int = int(anio)
        anio_formateado = anio.zfill(2)  # Asegurar 2 dígitos: '25' -> '25', '5' -> '05'
        if anio_int <= 50:
            anio_completo = f"20{anio_formateado}"
        else:
            anio_completo = f"19{anio_formateado}"
        
        # Obtener el mes numérico
        mes_num = meses.get(mes_abrev, mes_abrev)
        
        # Formar fecha en formato ISO: YYYY-MM-DD
        fecha_iso = f"{anio_completo}-{mes_num}-{dia}"
        
        return fecha_iso
    except Exception as e:
        print(f"Error al convertir fecha '{fecha_str}': {e}")
        return fecha_str  # Retornar original si hay error

def normalizar_fechas_csv(carpeta_csv):
    """
    Normaliza las fechas en la columna 'FECHA CONEXIÓN' de todos los CSV en la carpeta
    """
    carpeta = Path(carpeta_csv)
    archivos_csv = list(carpeta.glob('*.csv'))
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en {carpeta_csv}")
        return
    
    print(f"Se encontraron {len(archivos_csv)} archivos CSV para procesar\n")
    
    for archivo in archivos_csv:
        try:
            print(f"Procesando: {archivo.name}")
            
            # Leer el CSV
            df = pd.read_csv(archivo, encoding='utf-8')
            
            # Verificar que existe la columna 'FECHA CONEXIÓN'
            if 'FECHA CONEXIÓN' not in df.columns:
                print(f"  ⚠️  Advertencia: No se encontró la columna 'FECHA CONEXIÓN' en {archivo.name}")
                continue
            
            # Convertir las fechas
            df['FECHA CONEXIÓN'] = df['FECHA CONEXIÓN'].astype(str).apply(convertir_fecha_espanol_a_iso)
            
            # Guardar el CSV actualizado (sobrescribir el original)
            df.to_csv(archivo, index=False, encoding='utf-8')
            
            print(f"  ✅ Fechas normalizadas correctamente ({len(df)} filas procesadas)")
            
        except Exception as e:
            print(f"  ❌ Error al procesar {archivo.name}: {e}")
    
    print(f"\n✅ Proceso completado. Se procesaron {len(archivos_csv)} archivos.")

if __name__ == "__main__":
    # Carpeta donde están los CSV
    carpeta_csv = "csv-zonas-wifi-separados-PruebaEdier"
    
    # Verificar que la carpeta existe
    if not os.path.exists(carpeta_csv):
        print(f"Error: La carpeta '{carpeta_csv}' no existe.")
    else:
        normalizar_fechas_csv(carpeta_csv)


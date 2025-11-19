import pandas as pd
import os
import re
from pathlib import Path

# ============================
# CONFIGURACIÓN
# ============================
CARPETA_CSV = "csv-zonas-wifi-separados-PruebaEdier"

# ============================
# DICCIONARIO DE NORMALIZACIÓN
# ============================
# Mapeo de nombres comunes a formato estándar
MAPEO_COLUMNAS = {
    # Fechas
    'FECHA CONEXIÓN': 'FECHA_CONEXION',
    'FECHA.CONEXION': 'FECHA_CONEXION',
    'FECHA_CONEXION': 'FECHA_CONEXION',
    'FECHA CONEXION': 'FECHA_CONEXION',
    
    # Porcentaje
    'PORCENTAJE USO': 'PORCENTAJE_USO',
    'PORCENTAJE.USO': 'PORCENTAJE_USO',
    'PORCENTAJE_USO': 'PORCENTAJE_USO',
    
    # Número de conexiones
    'NÚMERO CONEXIONES': 'NUMERO_CONEXIONES',
    'NUMERO.CONEXIONES': 'NUMERO_CONEXIONES',
    'NUMERO_CONEXIONES': 'NUMERO_CONEXIONES',
    'NÚMERO.CONEXIONES': 'NUMERO_CONEXIONES',
    
    # Usage
    'USAGE (KB)': 'USAGE_KB',
    'USAGE (kB)': 'USAGE_KB',
    'USAGE.KB': 'USAGE_KB',
    'USAGE_KB': 'USAGE_KB',
    'USAGE(KB)': 'USAGE_KB',
    'USAGE(kB)': 'USAGE_KB',
    
    # Otras columnas comunes
    'NOMBRE ZONA': 'NOMBRE_ZONA',
    'NOMBRE.ZONA': 'NOMBRE_ZONA',
    'NOMBRE_ZONA': 'NOMBRE_ZONA',
}

def normalizar_nombre_columna(nombre):
    """
    Normaliza un nombre de columna a formato estándar:
    - Todo en MAYÚSCULAS
    - Guión bajo en lugar de espacios, puntos, paréntesis
    - Sin tildes ni caracteres especiales
    """
    # Si está en el mapeo, usar ese valor
    if nombre in MAPEO_COLUMNAS:
        return MAPEO_COLUMNAS[nombre]
    
    # Convertir a mayúsculas
    nombre_normalizado = nombre.upper()
    
    # Reemplazar espacios, puntos, paréntesis por guión bajo
    nombre_normalizado = re.sub(r'[\s\.\(\)]+', '_', nombre_normalizado)
    
    # Remover tildes y caracteres especiales
    nombre_normalizado = nombre_normalizado.replace('Á', 'A')
    nombre_normalizado = nombre_normalizado.replace('É', 'E')
    nombre_normalizado = nombre_normalizado.replace('Í', 'I')
    nombre_normalizado = nombre_normalizado.replace('Ó', 'O')
    nombre_normalizado = nombre_normalizado.replace('Ú', 'U')
    nombre_normalizado = nombre_normalizado.replace('Ñ', 'N')
    
    # Limpiar guiones bajos múltiples
    nombre_normalizado = re.sub(r'_+', '_', nombre_normalizado)
    
    # Remover guiones bajos al inicio y final
    nombre_normalizado = nombre_normalizado.strip('_')
    
    return nombre_normalizado

def procesar_csv(archivo):
    """
    Procesa un archivo CSV normalizando los nombres de columnas.
    """
    try:
        nombre = os.path.basename(archivo)
        print(f"Procesando: {nombre}")
        
        # Leer CSV
        df = pd.read_csv(archivo, encoding='utf-8')
        
        # Guardar nombres originales
        columnas_originales = df.columns.tolist()
        
        # Normalizar nombres de columnas
        columnas_normalizadas = [normalizar_nombre_columna(col) for col in columnas_originales]
        
        # Crear diccionario de mapeo
        mapeo = dict(zip(columnas_originales, columnas_normalizadas))
        
        # Renombrar columnas
        df.rename(columns=mapeo, inplace=True)
        
        # Mostrar cambios si hay diferencias
        cambios = []
        for orig, nueva in mapeo.items():
            if orig != nueva:
                cambios.append(f"  '{orig}' → '{nueva}'")
        
        if cambios:
            print(f"  📝 Cambios realizados:")
            for cambio in cambios[:10]:  # Mostrar máximo 10 cambios
                print(cambio)
            if len(cambios) > 10:
                print(f"  ... y {len(cambios) - 10} cambios más")
        else:
            print(f"  ✅ Sin cambios necesarios")
        
        # Guardar archivo (sobrescribir original)
        df.to_csv(archivo, index=False, encoding='utf-8')
        
        print(f"  ✅ Guardado: {len(df)} filas, {len(df.columns)} columnas")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error al procesar {nombre}: {e}")
        return False

# ============================
# PROGRAMA PRINCIPAL
# ============================

def main():
    # Verificar que la carpeta existe
    if not os.path.exists(CARPETA_CSV):
        print(f"❌ Error: La carpeta '{CARPETA_CSV}' no existe.")
        return
    
    # Buscar archivos CSV
    carpeta = Path(CARPETA_CSV)
    archivos_csv = list(carpeta.glob('*.csv'))
    
    if not archivos_csv:
        print(f"❌ No se encontraron archivos CSV en '{CARPETA_CSV}'")
        return
    
    print(f"📁 Se encontraron {len(archivos_csv)} archivos CSV para procesar\n")
    
    # Procesar cada archivo
    exitosos = 0
    fallidos = 0
    
    for archivo in archivos_csv:
        if procesar_csv(archivo):
            exitosos += 1
        else:
            fallidos += 1
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"✅ Procesamiento completado:")
    print(f"   Archivos procesados exitosamente: {exitosos}")
    if fallidos > 0:
        print(f"   Archivos con errores: {fallidos}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


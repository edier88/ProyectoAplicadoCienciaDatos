import pandas as pd
import os
from pathlib import Path

def normalizar_porcentaje(valor):
    """
    Unifica todos los valores al formato con % y corrige valores anómalos.
    - Los valores que ya tienen %:
      * Si el valor es > 10000, se corrige dividiendo por 100 (ej: '26127300,00%' -> '261273,00%')
      * Si el valor es <= 10000, se mantiene sin cambios (ej: '65,10%' -> '65,10%')
    - Los valores sin %:
      * Si el valor es <= 1, se asume que es decimal y se multiplica por 100 (ej: '0,36293183' -> '36,29%')
      * Si el valor es > 1, se asume que es decimal y se multiplica por 100 (ej: '1,36293183' -> '136,29%')
    """
    # Convertir a string si no lo es
    valor_str = str(valor).strip()
    
    # Si el valor ya tiene %, verificar si es anómalo
    if '%' in valor_str:
        try:
            # Remover el símbolo % y convertir a float
            valor_sin_porcentaje = valor_str.replace('%', '').strip()
            valor_sin_coma = valor_sin_porcentaje.replace(',', '.')
            valor_float = float(valor_sin_coma)
            
            # Si el valor es mayor a 10000, probablemente fue multiplicado incorrectamente
            # Lo dividimos por 100 para corregirlo
            if valor_float > 10000:
                valor_corregido = valor_float / 100
                valor_formateado = f"{valor_corregido:.2f}".replace('.', ',')
                return f"{valor_formateado}%"
            else:
                # El valor parece correcto, retornarlo sin cambios
                return valor_str
        except:
            # Si hay error al procesar, retornar original
            return valor_str
    
    try:
        # Reemplazar coma por punto para convertir a float
        valor_sin_coma = valor_str.replace(',', '.')
        
        # Convertir a float
        valor_float = float(valor_sin_coma)
        
        # Si el valor es mayor a 1, asumimos que ya es un porcentaje (solo agregar %)
        # Si el valor es <= 1, asumimos que es decimal (multiplicar por 100)
        if valor_float > 1:
            # Es decimal, multiplicar por 100 para convertir a porcentaje
            valor_porcentaje = valor_float * 100
            
            # Formatear con 2 decimales y reemplazar punto por coma
            valor_formateado = f"{valor_porcentaje:.2f}".replace('.', ',')
            
            # Agregar el símbolo %
            valor_final = f"{valor_formateado}%"
        else:
            # Es decimal, multiplicar por 100 para convertir a porcentaje
            valor_porcentaje = valor_float * 100
            
            # Formatear con 2 decimales y reemplazar punto por coma
            valor_formateado = f"{valor_porcentaje:.2f}".replace('.', ',')
            
            # Agregar el símbolo %
            valor_final = f"{valor_formateado}%"
        
        return valor_final
        
    except Exception as e:
        print(f"  ⚠️  Error al normalizar '{valor_str}': {e}")
        return valor_str  # Retornar original si hay error

def normalizar_porcentaje_uso_csv(carpeta_csv):
    """
    Unifica la columna 'PORCENTAJE USO' en todos los CSV al formato con %.
    - Corrige valores anómalos (con % > 10000) dividiéndolos por 100
    - Los valores con % normales se mantienen sin cambios
    - Los valores sin % se convierten a porcentaje según su valor (decimal o ya porcentaje)
    """
    carpeta = Path(carpeta_csv)
    archivos_csv = list(carpeta.glob('*.csv'))
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en {carpeta_csv}")
        return
    
    print(f"Se encontraron {len(archivos_csv)} archivos CSV para procesar\n")
    
    total_procesados = 0
    total_con_porcentaje = 0
    total_sin_porcentaje = 0
    total_corregidos = 0
    
    for archivo in archivos_csv:
        try:
            print(f"Procesando: {archivo.name}")
            
            # Leer el CSV
            df = pd.read_csv(archivo, encoding='utf-8')
            
            # Verificar que existe la columna 'PORCENTAJE USO'
            if 'PORCENTAJE USO' not in df.columns:
                print(f"  ⚠️  Advertencia: No se encontró la columna 'PORCENTAJE USO' en {archivo.name}")
                continue
            
            # Contar valores antes de normalizar
            valores_con_porcentaje = df['PORCENTAJE USO'].astype(str).str.contains('%', na=False).sum()
            valores_sin_porcentaje = len(df) - valores_con_porcentaje
            
            # Contar valores anómalos (con % > 10000)
            valores_anomalos = 0
            for val in df['PORCENTAJE USO'].astype(str):
                if '%' in val:
                    try:
                        valor_num = float(val.replace('%', '').replace(',', '.'))
                        if valor_num > 10000:
                            valores_anomalos += 1
                    except:
                        pass
            
            # Normalizar todos los valores (unificar a formato con % y corregir anómalos)
            df['PORCENTAJE USO'] = df['PORCENTAJE USO'].apply(normalizar_porcentaje)
            
            # Guardar el CSV actualizado (sobrescribir el original)
            df.to_csv(archivo, index=False, encoding='utf-8')
            
            mensaje = f"  ✅ Procesado: {valores_con_porcentaje} con %, {valores_sin_porcentaje} sin %"
            if valores_anomalos > 0:
                mensaje += f", {valores_anomalos} corregidos"
            mensaje += f" ({len(df)} filas totales)"
            print(mensaje)
            
            total_procesados += len(df)
            total_con_porcentaje += valores_con_porcentaje
            total_sin_porcentaje += valores_sin_porcentaje
            total_corregidos += valores_anomalos
            
        except Exception as e:
            print(f"  ❌ Error al procesar {archivo.name}: {e}")
    
    print(f"\n✅ Proceso completado.")
    print(f"   Total de filas procesadas: {total_procesados}")
    print(f"   Valores que ya tenían % (sin cambios): {total_con_porcentaje}")
    print(f"   Valores convertidos a %: {total_sin_porcentaje}")
    if total_corregidos > 0:
        print(f"   Valores anómalos corregidos: {total_corregidos}")

if __name__ == "__main__":
    # Carpeta donde están los CSV
    carpeta_csv = "csv-zonas-wifi-separados-PruebaEdier"
    
    # Verificar que la carpeta existe
    if not os.path.exists(carpeta_csv):
        print(f"Error: La carpeta '{carpeta_csv}' no existe.")
    else:
        normalizar_porcentaje_uso_csv(carpeta_csv)


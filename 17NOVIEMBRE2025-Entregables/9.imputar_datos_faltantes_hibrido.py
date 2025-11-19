import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURACIÓN
# ============================
CARPETA_CSV = "csv-zonas-wifi-separados-PruebaEdier"

# ============================
# FUNCIONES DE IMPUTACIÓN
# ============================

def convertir_porcentaje_a_float(valor):
    """Convierte porcentaje con formato '424,40%' a float 424.40"""
    if pd.isna(valor):
        return np.nan
    try:
        valor_str = str(valor).replace('%', '').replace(',', '.')
        return float(valor_str)
    except:
        return np.nan

def convertir_float_a_porcentaje(valor):
    """Convierte float 424.40 a formato '424,40%'"""
    if pd.isna(valor):
        return np.nan
    try:
        return f"{valor:.2f}".replace('.', ',') + '%'
    except:
        return np.nan

def imputar_usage_kb_logaritmico_semantico(df, columna_fecha):
    """
    Imputa USAGE_KB usando interpolación logarítmica + regresión semántica.
    Mejorado para manejar grandes bloques de datos faltantes.
    """
    # Crear copia para trabajar
    df_work = df.copy()
    
    # Convertir USAGE_KB a numérico si es necesario
    if df_work['USAGE_KB'].dtype == 'object':
        df_work['USAGE_KB'] = pd.to_numeric(df_work['USAGE_KB'], errors='coerce')
    
    # Identificar valores faltantes
    mask_faltantes = df_work['USAGE_KB'].isna()
    n_faltantes = mask_faltantes.sum()
    
    if n_faltantes == 0:
        return df_work['USAGE_KB'].values
    
    # Paso 1: Interpolación logarítmica temporal con extrapolación
    # Aplicar logaritmo a los valores existentes (evitar log(0))
    df_work['USAGE_KB_LOG'] = np.log1p(df_work['USAGE_KB'])  # log1p(x) = log(1+x)
    
    # Interpolación con pandas (maneja mejor grandes huecos)
    valores_interp_log = df_work['USAGE_KB_LOG'].interpolate(
        method='linear',
        limit_direction='both',  # Interpolar hacia adelante y atrás
        limit=None  # Sin límite de valores consecutivos
    )
    
    # Si aún hay NaN al inicio, usar forward fill
    valores_interp_log = valores_interp_log.ffill()
    
    # Si aún hay NaN al final, usar backward fill
    valores_interp_log = valores_interp_log.bfill()
    
    # Si aún hay NaN, usar la media de los valores existentes
    if valores_interp_log.isna().any():
        media_log = df_work['USAGE_KB_LOG'].mean()
        valores_interp_log = valores_interp_log.fillna(media_log)
    
    # Convertir de vuelta a escala normal
    valores_interp_log = np.expm1(valores_interp_log.values)  # expm1(x) = exp(x) - 1
    
    # Paso 2: Regresión semántica para ajustar
    # Preparar variables semánticas (incluir más contexto)
    variables_semanticas = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO']
    
    # Agregar variables temporales si están disponibles
    if columna_fecha in df_work.columns:
        df_work['MES'] = df_work[columna_fecha].dt.month
        df_work['DIA_MES'] = df_work[columna_fecha].dt.day
        variables_semanticas.extend(['MES', 'DIA_MES'])
    
    # Filtrar solo las que existen
    variables_semanticas = [v for v in variables_semanticas if v in df_work.columns]
    X_semantico = df_work[variables_semanticas].copy()
    
    # Rellenar NaN en variables semánticas con la moda
    for col in variables_semanticas:
        if X_semantico[col].isna().any():
            moda = X_semantico[col].mode()[0] if len(X_semantico[col].mode()) > 0 else 0
            X_semantico[col].fillna(moda, inplace=True)
    
    # Datos para entrenar (solo valores existentes)
    mask_entrenar = ~df_work['USAGE_KB'].isna()
    
    if mask_entrenar.sum() >= 5:  # Reducido a mínimo 5 muestras
        X_train = X_semantico[mask_entrenar].values
        y_train = df_work.loc[mask_entrenar, 'USAGE_KB'].values
        
        # Entrenar modelo de regresión
        try:
            # Ajustar parámetros para datasets pequeños
            n_estimators = min(50, max(10, mask_entrenar.sum() // 2))
            modelo = RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=42, 
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1
            )
            modelo.fit(X_train, y_train)
            
            # Predecir para todos los valores (incluyendo faltantes)
            X_all = X_semantico.values
            predicciones_semanticas = modelo.predict(X_all)
            
            # Asegurar que las predicciones no sean negativas
            predicciones_semanticas = np.maximum(predicciones_semanticas, 0)
            
            # Combinar: 60% interpolación logarítmica + 40% regresión semántica
            # (Más peso a regresión semántica para grandes huecos)
            valores_finales = df_work['USAGE_KB'].values.copy()
            valores_finales[mask_faltantes] = (
                0.6 * valores_interp_log[mask_faltantes] + 
                0.4 * predicciones_semanticas[mask_faltantes]
            )
        except Exception as e:
            # Si falla la regresión, usar solo interpolación logarítmica
            valores_finales = valores_interp_log
    else:
        # Si no hay suficientes datos, usar solo interpolación logarítmica
        valores_finales = valores_interp_log
    
    # Asegurar que no haya valores negativos
    valores_finales = np.maximum(valores_finales, 0)
    
    # Redondear a enteros (los valores originales son enteros con .0)
    valores_finales = np.round(valores_finales).astype(float)
    
    return valores_finales

def imputar_numero_conexiones_temporal_suavizado(df, columna_fecha):
    """
    Imputa NUMERO_CONEXIONES usando interpolación temporal con suavizado.
    Mejorado para manejar grandes bloques de datos faltantes.
    """
    # Crear copia para trabajar
    df_work = df.copy()
    
    # Convertir a numérico si es necesario
    if df_work['NUMERO_CONEXIONES'].dtype == 'object':
        df_work['NUMERO_CONEXIONES'] = pd.to_numeric(df_work['NUMERO_CONEXIONES'], errors='coerce')
    
    # Identificar valores faltantes
    mask_faltantes = df_work['NUMERO_CONEXIONES'].isna()
    n_faltantes = mask_faltantes.sum()
    
    if n_faltantes == 0:
        return df_work['NUMERO_CONEXIONES'].values
    
    # Paso 1: Interpolación temporal (lineal) con extrapolación
    valores_interp = df_work['NUMERO_CONEXIONES'].interpolate(
        method='linear',
        limit_direction='both',  # Interpolar hacia adelante y atrás
        limit=None  # Sin límite de valores consecutivos
    )
    
    # Si aún hay NaN al inicio, usar forward fill
    valores_interp = valores_interp.ffill()
    
    # Si aún hay NaN al final, usar backward fill
    valores_interp = valores_interp.bfill()
    
    # Si aún hay NaN, usar la media de los valores existentes
    if valores_interp.isna().any():
        media = df_work['NUMERO_CONEXIONES'].mean()
        valores_interp = valores_interp.fillna(media)
    
    valores_interp = valores_interp.values
    
    # Paso 2: Suavizado con filtro Savitzky-Golay
    valores_existentes = ~df_work['NUMERO_CONEXIONES'].isna()
    n_existentes = valores_existentes.sum()
    
    if n_existentes >= 5 and len(valores_interp) >= 5:  # Mínimo para aplicar suavizado
        try:
            # Aplicar suavizado solo si hay suficientes puntos
            window_length = min(5, len(valores_interp) if len(valores_interp) % 2 == 1 else len(valores_interp) - 1)
            if window_length >= 3:
                valores_suavizados = savgol_filter(
                    valores_interp,
                    window_length=window_length,
                    polyorder=min(2, window_length - 1)
                )
            else:
                valores_suavizados = valores_interp
        except:
            valores_suavizados = valores_interp
    else:
        valores_suavizados = valores_interp
    
    # Asegurar que no haya valores negativos y que sean enteros
    valores_suavizados = np.maximum(valores_suavizados, 0)
    valores_suavizados = np.round(valores_suavizados).astype(float)
    
    return valores_suavizados

def calcular_porcentaje_uso(df, usage_kb_imputado, numero_conexiones_imputado):
    """
    Calcula PORCENTAJE_USO matemáticamente basado en USAGE_KB y NUMERO_CONEXIONES.
    
    Fórmula: PORCENTAJE_USO = (USAGE_KB / (NUMERO_CONEXIONES * capacidad_estimada)) * 100
    
    Donde capacidad_estimada se estima de los datos históricos.
    """
    df_work = df.copy()
    
    # Convertir valores existentes de PORCENTAJE_USO a float
    porcentajes_existentes = df_work['PORCENTAJE_USO'].apply(convertir_porcentaje_a_float)
    
    # Estimar capacidad promedio basada en datos históricos
    # Si tenemos USAGE_KB y PORCENTAJE_USO, podemos estimar la capacidad
    mask_validos = (~pd.isna(usage_kb_imputado)) & (~pd.isna(numero_conexiones_imputado)) & (~pd.isna(porcentajes_existentes))
    mask_validos = mask_validos & (numero_conexiones_imputado > 0) & (porcentajes_existentes > 0)
    
    if mask_validos.sum() > 0:
        # Estimar capacidad: capacidad = USAGE_KB / (PORCENTAJE_USO / 100)
        capacidades_estimadas = usage_kb_imputado[mask_validos] / (porcentajes_existentes[mask_validos] / 100)
        capacidad_promedio = capacidades_estimadas.mean()
    else:
        # Si no hay datos históricos, usar una capacidad estimada conservadora
        # Basada en valores típicos de WiFi (ej: 1000 KB por conexión)
        capacidad_promedio = 1000.0
    
    # Calcular porcentaje de uso para todos los valores
    porcentajes_calculados = np.zeros(len(df_work))
    
    for i in range(len(df_work)):
        usage = usage_kb_imputado[i]
        conexiones = numero_conexiones_imputado[i]
        
        if not pd.isna(usage) and not pd.isna(conexiones) and conexiones > 0:
            # Fórmula: PORCENTAJE = (USAGE_KB / (NUMERO_CONEXIONES * capacidad)) * 100
            capacidad_total = conexiones * capacidad_promedio
            if capacidad_total > 0:
                porcentaje = (usage / capacidad_total) * 100
                porcentajes_calculados[i] = porcentaje
            else:
                porcentajes_calculados[i] = 0
        else:
            porcentajes_calculados[i] = np.nan
    
    # Si hay valores históricos, usar promedio ponderado (70% calculado, 30% histórico)
    if mask_validos.sum() > 0:
        for i in range(len(df_work)):
            if mask_validos[i]:
                porcentaje_historico = porcentajes_existentes.iloc[i] if hasattr(porcentajes_existentes, 'iloc') else porcentajes_existentes.values[i]
                if not pd.isna(porcentaje_historico):
                    porcentajes_calculados[i] = 0.7 * porcentajes_calculados[i] + 0.3 * porcentaje_historico
    
    return porcentajes_calculados

def procesar_csv(archivo):
    """
    Procesa un CSV aplicando imputación híbrida a las columnas objetivo.
    """
    try:
        nombre = os.path.basename(archivo)
        print(f"\nProcesando: {nombre}")
        
        # Leer CSV
        df = pd.read_csv(archivo, encoding='utf-8')
        
        # Detectar columna de fecha
        columna_fecha = None
        posibles_nombres = ['FECHA_CONEXION', 'FECHA.CONEXION', 'FECHA CONEXIÓN', 'FECHA CONEXION']
        for nombre_posible in posibles_nombres:
            if nombre_posible in df.columns:
                columna_fecha = nombre_posible
                break
        
        if columna_fecha is None:
            print(f"  ❌ No se encontró columna de fecha")
            return False
        
        # Convertir fecha a datetime
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')
        df = df.sort_values(columna_fecha).reset_index(drop=True)
        
        # Verificar columnas necesarias
        columnas_requeridas = ['USAGE_KB', 'NUMERO_CONEXIONES', 'PORCENTAJE_USO']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if columnas_faltantes:
            print(f"  ❌ Faltan columnas: {', '.join(columnas_faltantes)}")
            return False
        
        # Contar valores faltantes antes
        faltantes_antes = {
            'USAGE_KB': df['USAGE_KB'].isna().sum(),
            'NUMERO_CONEXIONES': df['NUMERO_CONEXIONES'].isna().sum(),
            'PORCENTAJE_USO': df['PORCENTAJE_USO'].isna().sum()
        }
        
        # 1. Imputar USAGE_KB (interpolación logarítmica + regresión semántica)
        print(f"  📊 Imputando USAGE_KB... ({faltantes_antes['USAGE_KB']} faltantes)")
        df['USAGE_KB'] = imputar_usage_kb_logaritmico_semantico(df, columna_fecha)
        
        # 2. Imputar NUMERO_CONEXIONES (interpolación temporal + suavizado)
        print(f"  📊 Imputando NUMERO_CONEXIONES... ({faltantes_antes['NUMERO_CONEXIONES']} faltantes)")
        df['NUMERO_CONEXIONES'] = imputar_numero_conexiones_temporal_suavizado(df, columna_fecha)
        
        # 3. Calcular PORCENTAJE_USO (derivación matemática)
        print(f"  📊 Calculando PORCENTAJE_USO... ({faltantes_antes['PORCENTAJE_USO']} faltantes)")
        porcentajes_calculados = calcular_porcentaje_uso(
            df,
            df['USAGE_KB'].values,
            df['NUMERO_CONEXIONES'].values
        )
        
        # Convertir porcentajes a formato string con %
        porcentajes_formateados = [convertir_float_a_porcentaje(p) if not pd.isna(p) else np.nan 
                                   for p in porcentajes_calculados]
        df['PORCENTAJE_USO'] = porcentajes_formateados
        
        # Asegurar formato correcto: USAGE_KB y NUMERO_CONEXIONES como enteros (float con .0)
        # Esto asegura que se guarden como 3709424.0 en lugar de 3709424.723179051
        df['USAGE_KB'] = df['USAGE_KB'].round(0).astype(float)
        df['NUMERO_CONEXIONES'] = df['NUMERO_CONEXIONES'].round(0).astype(float)
        
        # Guardar archivo
        df.to_csv(archivo, index=False, encoding='utf-8', float_format='%.1f')
        
        # Estadísticas finales
        faltantes_despues = {
            'USAGE_KB': pd.isna(df['USAGE_KB']).sum() if df['USAGE_KB'].dtype != 'object' else 0,
            'NUMERO_CONEXIONES': pd.isna(df['NUMERO_CONEXIONES']).sum() if df['NUMERO_CONEXIONES'].dtype != 'object' else 0,
            'PORCENTAJE_USO': df['PORCENTAJE_USO'].isna().sum()
        }
        
        print(f"  ✅ Completado:")
        print(f"     USAGE_KB: {faltantes_antes['USAGE_KB']} → {faltantes_despues['USAGE_KB']} faltantes")
        print(f"     NUMERO_CONEXIONES: {faltantes_antes['NUMERO_CONEXIONES']} → {faltantes_despues['NUMERO_CONEXIONES']} faltantes")
        print(f"     PORCENTAJE_USO: {faltantes_antes['PORCENTAJE_USO']} → {faltantes_despues['PORCENTAJE_USO']} faltantes")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error procesando {nombre}: {e}")
        import traceback
        traceback.print_exc()
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
    
    print(f"📁 Se encontraron {len(archivos_csv)} archivos CSV para procesar")
    print(f"\n🔧 Técnicas de imputación:")
    print(f"   • USAGE_KB: Interpolación logarítmica + Regresión semántica")
    print(f"   • NUMERO_CONEXIONES: Interpolación temporal + Suavizado")
    print(f"   • PORCENTAJE_USO: Derivación matemática")
    
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


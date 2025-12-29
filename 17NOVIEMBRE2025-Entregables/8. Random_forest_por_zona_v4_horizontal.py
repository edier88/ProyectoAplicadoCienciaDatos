# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.datasets import fetch_dataset

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from skforecast.plot import set_dark_theme

# Modelado y Forecasting
# ==============================================================================
import sklearn
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import skforecast
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.utils import save_forecaster, load_forecaster
from skforecast.metrics import calculate_coverage
from skforecast.plot import plot_prediction_intervals
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import shap

import math
import glob
import os
import joblib
from pathlib import Path

GRAF_DIR = "Random_Forest_Graficas_TrainTest"
os.makedirs(GRAF_DIR, exist_ok=True)

MODELOS_DIR = Path("Random_Forest_Modelos_Guardados")
os.makedirs(MODELOS_DIR, exist_ok=True)

ORIGEN = "csv-zonas-wifi-separados-man/"
#DESTINO_METRICAS = "Random_Forest_Metricas"
DESTINO_METRICAS = Path("Random_Forest_Metricas")
os.makedirs(DESTINO_METRICAS, exist_ok=True)

#carpeta = os.path.join(os.path.dirname(__file__), "csv-zonas-wifi-1AP-todas-las-columnas")
#print(carpeta)

archivos = glob.glob(os.path.join(ORIGEN, "*.csv"))

#df_errors = pd.DataFrame(columns=["Zona", "y_true", "y_pred", "MAPE", "error_abs", "error_relativo"])
df_errors = pd.DataFrame(columns=["Zona", "Modelo", "MAPE_Base", "MAPE_Optimizado", "MAPE(%)_Base", "MAPE(%)_Optimizado", "MAE_Base", "MAE_Optimizado", "RMSE_Base", "RMSE_Optimizado", "R2_Base", "R2_Optimizado"])
#df_errors_ajustado = pd.DataFrame(columns=["Zona", "Tecnica", "MAPE", "MAPE(%)", "MAE", "RMSE", "R2"])
mape_percent = 0

for archivo in archivos:
    nombre_zona = os.path.basename(archivo)
    nombre_zona_recortado = nombre_zona[:-4]
    print(f"\nProcesando: {nombre_zona}")
    print(f"\nProcesando RECORTADO: {nombre_zona_recortado}")
    
    #df = pd.read_csv(os.path.join(carpeta,"001_ZW Parque Ingenio-test2.csv"))
    df = pd.read_csv(archivo)
    
    # Preparación del dato
    # ==============================================================================
    df['FECHA_CONEXION'] = pd.to_datetime(df['FECHA_CONEXION'], format='%Y-%m-%d')
    df = df.set_index('FECHA_CONEXION')
    df = df.asfreq('D') # Fechas con frecuencia diario. En caso de que falte algún día se crea y todas las demás variables se ponen NaN
    
    df = df.sort_index() # En caso de que esten las fechas desorganizadas se organizan de forma ascendente
    
    # Hacer método para detectar si hay datos faltantes y no proseguir con el modelado
    df.isnull().any(axis=1).mean()
    
    rows_with_na = df[df.isna().any(axis=1)]
    print("\nRows with any NA values:")
    print(rows_with_na)
    
    rows, columns = df.shape
    print(rows)
    
    df['DIA_SEMANA'] = df['DIA_SEMANA'].astype('Int64')
    df['LABORAL'] = df['LABORAL'].astype('Int64')
    df['FIN_DE_SEMANA'] = df['FIN_DE_SEMANA'].astype('Int64')
    df['FESTIVO'] = df['FESTIVO'].astype('Int64')
    df['PORCENTAJE_USO'] = df['PORCENTAJE_USO'].astype('Float64')
    df['NUMERO_CONEXIONES'] = df['NUMERO_CONEXIONES'].astype('Float64')
    df['USAGE_KB'] = df['USAGE_KB'].astype('Float64')






    # Deepseek

    # DEFINIR VARIABLES
    target_var = 'USAGE_KB'
    exog_vars_numericas = ['PORCENTAJE_USO', 'NUMERO_CONEXIONES']
    exog_vars_categoricas = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO']
    exog_variables = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO', 'NUMERO_CONEXIONES']
    lags_exogenos = 10  # Lags para variables exógenas
    lags_target = 10    # Lags para variable target (USAGE_KB)

    print(f"Procesando dataset con {len(df)} filas")

    # FUNCIÓN CORREGIDA: Solo crea lags para exógenas
    def crear_lags_exogenos(df, exog_variables, lags):
        """Crea lags solo para variables exógenas numéricas"""
        df_con_lags = df.copy()
        
        """
        for var in exog_vars:
            for lag in range(1, lags + 1):
                df_con_lags[f'{var}_lag_{lag}'] = df[var].shift(lag)
        
        # Eliminar filas con NaN
        df_con_lags = df_con_lags.iloc[lags:].copy()
        return df_con_lags
        """

        for var in exog_variables:
            print(f"  {var}: lags 1 a {lags}")
            for lag in range(1, lags + 1):
                col_name = f'{var}_lag_{lag}'
                df_con_lags[col_name] = df[var].shift(lag)
    
        # Eliminar filas con NaN
        df_con_lags = df_con_lags.iloc[lags:].copy()
    
        print(f"Dataset original: {len(df)} filas")
        print(f"Dataset con lags: {len(df_con_lags)} filas")
        print(f"Se eliminaron {lags} filas iniciales con NaN")
        
        return df_con_lags

    # FUNCIÓN para dividir
    def dividir_train_test(df, test_size=0.2):
        n_total = len(df)
        n_test = int(n_total * test_size)
        n_train = n_total - n_test
        
        df_train = df.iloc[:n_train].copy()
        df_test = df.iloc[n_train:].copy()
        
        print(f"\nDivisión train/test:")
        print(f"  Train: {len(df_train)} filas")
        print(f"  Test:  {len(df_test)} filas")
        
        # Verificar continuidad
        if hasattr(df_train.index, '__len__') and hasattr(df_test.index, '__len__'):
            if len(df_train) > 0 and len(df_test) > 0:
                ultimo_train = df_train.index[-1]
                primero_test = df_test.index[0]
                diferencia = (primero_test - ultimo_train).days
                
                if diferencia == 1:
                    print(f"  ✅ Continuidad perfecta (+1 día)")
                elif diferencia > 1:
                    print(f"  ⚠️  Brecha de {diferencia} días")
                else:
                    print(f"  ⚠️  Solapamiento de {abs(diferencia)} días")
        
        return df_train, df_test

    # 1. Crear lags solo para exógenas
    df_con_lags = crear_lags_exogenos(df, exog_variables, lags_exogenos)
    print(f"Dataset después de crear lags: {len(df_con_lags)} filas")

    # 2. Dividir
    df_train, df_test = dividir_train_test(df_con_lags, test_size=0.2)

    # 3. Definir TODAS las variables exógenas
    #exog_variables = exog_vars_categoricas + exog_vars_numericas
    
    # 3. Definir TODAS las variables de entrada (originales + lags)
    # Primero las variables originales
    todas_variables_entrada = exog_variables.copy()

    # Luego añadir todos los lags
    for var in exog_variables:
        for lag in range(1, lags_exogenos + 1):
            todas_variables_entrada.append(f'{var}_lag_{lag}')

    print(f"\nVariables de entrada totales: {len(exog_variables)}")
    print("Ejemplo de variables:", exog_variables[:8], "...")

    # 4. Crear y entrenar forecaster
    # NOTA: Usa 'estimator' en lugar de 'regressor' (como sugiere el warning)
    forecaster = ForecasterRecursive(
        regressor=RandomForestRegressor(random_state=123),
        lags=lags_target  # Esto creará USAGE_KB_lag_1 a USAGE_KB_lag_10 automáticamente
    )

    print(f"\nEntrenando modelo...")
    print(f"  Variable objetivo: {target_var}")
    print(f"  Lags de {target_var}: {lags_target}")
    print(f"  Variables exógenas totales: {len(todas_variables_entrada)}")

    forecaster.fit(
        y=df_train[target_var],
        exog=df_train[todas_variables_entrada]
    )

    # 5. Verificar estructura
    print(f"\n✅ Modelo entrenado exitosamente")
    print(f"   Features totales: {forecaster.regressor.n_features_in_}")
    print(f"   Lags configurados: {forecaster.lags}")
    print(f"   Window size: {forecaster.window_size}")
    print(f"   exog names in: {forecaster.exog_names_in_}")

    # Verificar que tenemos todas las variables esperadas
    variables_esperadas = lags_target + len(todas_variables_entrada)
    print(f"\n🔍 Verificación:")
    print(f"   Lags target: {lags_target}")
    print(f"   Variables exógenas: {len(todas_variables_entrada)}")
    print(f"   Total esperado: {variables_esperadas}")
    print(f"   Total en modelo: {forecaster.regressor.n_features_in_}")

    if forecaster.regressor.n_features_in_ == variables_esperadas:
        print("   ✅ Coincide perfectamente")
    else:
        print(f"   ⚠️  Discrepancia: diferencia de {abs(forecaster.regressor.n_features_in_ - variables_esperadas)}")

    # 6. Hacer predicciones
    print(f"\nHaciendo predicciones...")
    predictions = forecaster.predict(
        steps=len(df_test),
        exog=df_test[todas_variables_entrada]
    )

    # 7. Evaluar
    mae = mean_absolute_error(df_test[target_var], predictions)
    mape = mean_absolute_percentage_error(df_test[target_var], predictions)

    print(f"\n📊 RESULTADOS:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.4f} ({mape*100:.2f}%)")

    # 8. Comparar con modelo sin lags de exógenas
    print(f"\n🔍 Comparación:")
    #print(f"  Modelo ANTERIOR (sin lags exógenos):")
    #print(f"    - Features: {lags_target} + {len(exog_vars_categoricas + exog_vars_numericas)} = {lags_target + len(exog_vars_categoricas + exog_vars_numericas)}")
    print(f"  Modelo NUEVO (con lags exógenos):")
    print(f"    - Features: {forecaster.regressor.n_features_in_}")
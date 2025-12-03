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
from sklearn.pipeline import Pipeline
import shap

import math
import glob
import os
from pathlib import Path

GRAF_DIR = "SVR_Graficas"
os.makedirs(GRAF_DIR, exist_ok=True)

ORIGEN = "csv-zonas-wifi-de-pamartin/"
#DESTINO_METRICAS = "Random_Forest_Metricas"
DESTINO_METRICAS = Path("SVR_Metricas_Pipeline")
os.makedirs(DESTINO_METRICAS, exist_ok=True)

#carpeta = os.path.join(os.path.dirname(__file__), "csv-zonas-wifi-1AP-todas-las-columnas")
#print(carpeta)

archivos = glob.glob(os.path.join(ORIGEN, "*.csv"))

#df_errors = pd.DataFrame(columns=["Zona", "y_true", "y_pred", "MAPE", "error_abs", "error_relativo"])
df_errors = pd.DataFrame(columns=["Zona", "MAPE", "MAPE(%)", "MAE", "RMSE", "R2"])
df_errors_ajustado = pd.DataFrame(columns=["Zona", "MAPE", "MAPE(%)", "MAE", "RMSE", "R2"])
mape_percent = 0

for archivo in archivos:
    nombre_zona = os.path.basename(archivo)
    print(f"\nProcesando: {nombre_zona}")
    
    #df = pd.read_csv(os.path.join(carpeta,"001_ZW Parque Ingenio-test2.csv"))
    df = pd.read_csv(archivo)
    
    # ==============================================================================
    # Preparación del dataset
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
    
    # Conversion de los números a los tipos adecuados
    df['DIA_SEMANA'] = df['DIA_SEMANA'].astype('Int64')
    df['LABORAL'] = df['LABORAL'].astype('Int64')
    df['FIN_DE_SEMANA'] = df['FIN_DE_SEMANA'].astype('Int64')
    df['FESTIVO'] = df['FESTIVO'].astype('Int64')
    df['PORCENTAJE_USO'] = df['PORCENTAJE_USO'].astype('Float64')
    df['NUMERO_CONEXIONES'] = df['NUMERO_CONEXIONES'].astype('Float64')
    df['USAGE_KB'] = df['USAGE_KB'].astype('Float64')
    
    # Separamos el dataset en 80% de train y 20% de test
    steps = rows*0.3 # 20% en test
    steps = math.floor(steps)
    print(f"Dataset separado con {steps} filas en test y {rows-steps} filas en la parte train")
    
    # Separación datos train-test
    # ==============================================================================
    df_train = df[:-steps].copy()
    df_test  = df[-steps:].copy()
    print(f"Fechas train : {df_train.index.min()} --- {df_train.index.max()}  (n={len(df_train)})")
    print(f"Fechas test  : {df_test.index.min()} --- {df_test.index.max()}  (n={len(df_test)})")
    
    # Variables de entrada para el modelo
    exog_variables = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO', 'NUMERO_CONEXIONES']
    # Define las variables exogenas (usa las versiones escaladas para variables continuas)
    #exog_variables_scaled = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO_scaled', 'NUMERO_CONEXIONES_scaled']

    """
    # Inicializar escaladores para cada variable
    scaler_usage = RobustScaler()
    scaler_conexiones = RobustScaler()
    scaler_porcentaje = RobustScaler()

    # Se escala la data de training
    df_train['USAGE_KB_scaled'] = scaler_usage.fit_transform(df_train[['USAGE_KB']])
    df_train['NUMERO_CONEXIONES_scaled'] = scaler_conexiones.fit_transform(df_train[['NUMERO_CONEXIONES']])
    df_train['PORCENTAJE_USO_scaled'] = scaler_porcentaje.fit_transform(df_train[['PORCENTAJE_USO']])

    # Se escala test aplicando los escaladores de train (Se usa "transform", no "fit_transform")
    df_test['USAGE_KB_scaled'] = scaler_usage.transform(df_test[['USAGE_KB']])
    df_test['NUMERO_CONEXIONES_scaled'] = scaler_conexiones.transform(df_test[['NUMERO_CONEXIONES']])
    df_test['PORCENTAJE_USO_scaled'] = scaler_porcentaje.transform(df_test[['PORCENTAJE_USO']])

    """

    # SVR pipeline with scaling
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('svr', SVR(kernel='rbf'))
    ])

    forecaster = ForecasterRecursive(
        regressor=pipeline,
        lags=10
    )

    # Define cross-validation
    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=int(len(df_train) * 0.6),  # Use 60% for initial training
        refit=False,
        fixed_train_size=False
    )

    # SVR parameter grid (note the double underscore for pipeline)
    param_grid = {
        'svr__C': [0.1, 1.0, 10.0],
        'svr__epsilon': [0.1, 0.01, 0.004],
        'svr__gamma': ['scale', 'auto', 0.5, 0.05],
        'svr__kernel': ['rbf']
    }

    # Grid search with ORIGINAL data
    resultados_grid = grid_search_forecaster(
        forecaster=forecaster,
        y=df_train['USAGE_KB'],  # Original scale
        exog=df_train[exog_variables],  # Original scale
        cv=cv,
        param_grid=param_grid,
        lags_grid=[6, 12, 18],
        metric='mean_absolute_percentage_error',
        return_best=True,
        n_jobs=1,
        verbose=False
    )

    resultados_grid.to_csv(DESTINO_METRICAS / f"grilla_pipeline_{nombre_zona}", index=False, encoding='utf-8')

    
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

GRAF_DIR = "RegresionLineal_Graficas_TrainTest"
os.makedirs(GRAF_DIR, exist_ok=True)

MODELOS_DIR = Path("RegresionLineal_Modelos_Guardados")
os.makedirs(MODELOS_DIR, exist_ok=True)

GRAF_FUTURAS_DIR = "RegresionLineal_Graficas_Futuras"
os.makedirs(GRAF_FUTURAS_DIR, exist_ok=True)

ORIGEN = "csv-zonas-wifi-separados-man/"
#DESTINO_METRICAS = "Random_Forest_Metricas"
DESTINO_METRICAS = Path("RegresionLineal_Metricas")
os.makedirs(DESTINO_METRICAS, exist_ok=True)

#carpeta = os.path.join(os.path.dirname(__file__), "csv-zonas-wifi-1AP-todas-las-columnas")
#print(carpeta)

archivos = glob.glob(os.path.join(ORIGEN, "*.csv"))

#df_errors = pd.DataFrame(columns=["Zona", "y_true", "y_pred", "MAPE", "error_abs", "error_relativo"])
#df_errors = pd.DataFrame(columns=["Zona", "MAPE", "MAE", "RMSE", "R2"])
df_errors = pd.DataFrame(columns=["Zona", "MAPE", "MAPE(%)", "MAE", "RMSE", "R2"])
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
    
    # Separamos el dataset en 70% de train y 30% de test
    steps = rows*0.2 # 20% en test
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
    exog_variables_scaled = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO_scaled', 'NUMERO_CONEXIONES_scaled']

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

    # Basic Linear Regression
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=10
    )

    forecaster.fit(
        y=df_train['USAGE_KB_scaled'],
        exog=df_train[exog_variables_scaled]
    )

    predictions_scaled = forecaster.predict(
        steps=steps,
        exog=df_test[exog_variables_scaled]
    )

    # Descaling the predictions
    predictions_final = pd.Series(
        scaler_usage.inverse_transform(predictions_scaled.values.reshape(-1, 1)).flatten(),
        index=predictions_scaled.index
    )

    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(
        y_true=df_test['USAGE_KB'],
        y_pred=predictions_final
    )
    mape_percent = mape*100
    mape = round(mape, 3)
    mape_percent = round(mape_percent, 3)
    print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")

    # Mean Absolute Error
    mae = mean_absolute_error(df_test['USAGE_KB'], predictions_final)
    mae = round(mae, 3)
    print(f"MAE: {mae:.2f}")

    # Root Mean Squared Error
    mse = mean_squared_error(df_test['USAGE_KB'], predictions_final)
    rmse = np.sqrt(mse)
    rmse = round(rmse, 3)
    print(f"RMSE: {rmse:.2f}")

    r2 = r2_score(df_test['USAGE_KB'], predictions_final)
    r2 = round(r2, 3)
    print(f"R-Cuadrado: {r2:.4f}")

    
    new_row = pd.DataFrame([{"Zona": nombre_zona, "MAPE": mape, "MAPE(%)": mape_percent, "MAE": mae, "RMSE": rmse, "R2": r2}])
    df_errors = pd.concat([df_errors, new_row], ignore_index=True)

    usage_kb_compared = pd.DataFrame({
        'USAGE_KB_predicho': predictions_final,
        'USAGE_KB_real': df_test['USAGE_KB']
    })
    difference = predictions_final - df_test['USAGE_KB']
    usage_kb_compared['error_absoluto'] = difference.abs()
    usage_kb_compared['error_relativo'] = usage_kb_compared['error_absoluto'] / usage_kb_compared['USAGE_KB_real']

    usage_kb_compared.to_csv(DESTINO_METRICAS / f"metricas_{nombre_zona}", index=False, encoding='utf-8')
    
    # Graficar serie temporal
    #print(plt.style.available)
    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(25, 4))
    plt.plot(df_train['USAGE_KB'], label="Train", linewidth=2)
    plt.plot(df_test['USAGE_KB'], label="Test", linewidth=2)
    plt.plot(predictions_final, label="Predicho", linewidth=2)
    plt.title(f"Regresion Lineal - {nombre_zona}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, f"{nombre_zona}_serie.png"), dpi=300)
    plt.close()



    df['USAGE_KB_scaled'] = scaler_usage.fit_transform(df[['USAGE_KB']])
    df['NUMERO_CONEXIONES_scaled'] = scaler_conexiones.fit_transform(df[['NUMERO_CONEXIONES']])
    df['PORCENTAJE_USO_scaled'] = scaler_porcentaje.fit_transform(df[['PORCENTAJE_USO']])

    # Basic Linear Regression
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=10
    )

    forecaster.fit(
        y=df['USAGE_KB_scaled'],
        exog=df[exog_variables_scaled]
    )

    # 5. Guardar TODO en un solo archivo
    modelo_completo = {
        'forecaster': forecaster,
        'scalers': {
            'scaler_usage': scaler_usage,          # Para USAGE_KB
            'scaler_conexiones': scaler_conexiones,  # Para NUMERO_CONEXIONES
            'scaler_porcentaje': scaler_porcentaje   # Para PORCENTAJE_USO
        },
        'variables_config': {
            'exog_variables_original': exog_variables,          # ['DIA_SEMANA', ...]
            'exog_variables_scaled': exog_variables_scaled,     # ['DIA_SEMANA', ..., '_scaled']
            'target_column': 'USAGE_KB',
            'scaled_target_column': 'USAGE_KB_scaled'
        },
        'scaler_info': {
            'scaler_usage_center': scaler_usage.center_.tolist(),
            'scaler_usage_scale': scaler_usage.scale_.tolist(),
            'scaler_conexiones_center': scaler_conexiones.center_.tolist(),
            'scaler_conexiones_scale': scaler_conexiones.scale_.tolist(),
            'scaler_porcentaje_center': scaler_porcentaje.center_.tolist(),
            'scaler_porcentaje_scale': scaler_porcentaje.scale_.tolist()
        },
        'metadata': {
            'zona': nombre_zona_recortado,
            'fecha_entrenamiento': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Regresion Lineal',
            'lags': 10
        }
    }

    # Guardar con joblib (mejor que pickle para objetos scikit-learn)
    joblib.dump(modelo_completo, MODELOS_DIR / f"RegresionLineal_{nombre_zona_recortado}.joblib")
    


df_errors.to_csv(DESTINO_METRICAS / "metricas.csv", index=False, encoding='utf-8')
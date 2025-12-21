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
    
    # Separamos el dataset en 80% de train y 20% de test
    steps = rows*0.2 # 20% en test
    steps = math.floor(steps)
    print(f"Dataset separado con {steps} filas en test y {rows-steps} filas en la parte train")
    
    # Separación datos train-test
    # ==============================================================================
    df_train = df[:-steps]
    df_test  = df[-steps:]
    print(f"Fechas train : {df_train.index.min()} --- {df_train.index.max()}  (n={len(df_train)})")
    print(f"Fechas test  : {df_test.index.min()} --- {df_test.index.max()}  (n={len(df_test)})")
    
    # Variables de entrada para el modelo
    exog_variables = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO', 'NUMERO_CONEXIONES']
    
    # Crea y entrena con variables exogenas. Usa ForecasterRecursive con RandomForest
    
    forecaster = ForecasterRecursive(
        regressor=RandomForestRegressor(random_state=123),
        lags=8
    )
    
    # Ajuste con variables exogenas
    forecaster.fit(
        y=df_train['USAGE_KB'],
        exog=df_train[exog_variables]  # Variables de entrada para alimentar el modelo
    )
    
    # Se realiza la predicción
    predictions = forecaster.predict(
        steps=steps,
        exog=df_test[exog_variables]
    )
    
    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(
        y_true=df_test['USAGE_KB'],
        y_pred=predictions
    )
    mape_base = round(mape, 6)
    mape_percent_base = mape_base*100
    mape_percent_base = round(mape_percent_base, 4)
    print(f"MAPE: {mape_base:.4f} ({mape_base*100:.2f}%)")

    # Mean Absolute Error
    mae_base = mean_absolute_error(df_test['USAGE_KB'], predictions)
    mae_base = round(mae_base, 3)
    print(f"MAE: {mae_base:.2f}")

    # Root Mean Squared Error
    mse_base = mean_squared_error(df_test['USAGE_KB'], predictions)
    rmse_base = np.sqrt(mse_base)
    rmse_base = round(rmse_base, 3)
    print(f"RMSE: {rmse_base:.2f}")

    r2_base = r2_score(df_test['USAGE_KB'], predictions)
    r2_base = round(r2_base, 3)
    print(f"R-Cuadrado: {r2_base:.4f}")

    #new_row = pd.DataFrame([{"Zona": nombre_zona, "Tecnica": "Sin Hiperparametros", "MAPE": mape, "MAPE(%)": mape_percent, "MAE": mae, "RMSE": rmse, "R2": r2}])
    #df_errors = pd.concat([df_errors, new_row], ignore_index=True)
    #df_errors.to_csv(DESTINO_METRICAS / "metricas.csv", index=False, encoding='utf-8')

    usage_kb_compared_scaled = pd.DataFrame({
        'USAGE_KB_predicho': predictions,
        'USAGE_KB_real': df_test['USAGE_KB']
    })
    difference = predictions - df_test['USAGE_KB']
    usage_kb_compared_scaled['error_absoluto'] = difference.abs()
    usage_kb_compared_scaled['error_relativo'] = usage_kb_compared_scaled['error_absoluto'] / usage_kb_compared_scaled['USAGE_KB_real']

    usage_kb_compared_scaled.to_csv(DESTINO_METRICAS / f"metricas_{nombre_zona}", index=False, encoding='utf-8')

    
    
    # ---------------------------------------------------------
    # Busqueda de Hiper-parámetros por zona:
    # ---------------------------------------------------------


    forecaster = ForecasterRecursive(
        regressor = RandomForestRegressor(random_state=123),
        lags      = 10 # Este valor será remplazado en el grid search
    )
    
    # particiones train y validacion
    cv = TimeSeriesFold(
        steps              = steps,
        initial_train_size = max(30, int(0.5 * len(df_train))),
        refit              = False,
        fixed_train_size   = False,
    )
    
    # Valores de lags para evaluar
    lags_grid = [10]
    
    # Valores a evaluar como hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 150, 250, 350],
        'max_depth': [5, 10, 20, 30, 40]
    }
    
    resultados_grid = grid_search_forecaster(
        forecaster  = forecaster,
        y           = df_train['USAGE_KB'],
        exog        = df_train[exog_variables],  # Variables exogenas
        cv          = cv,
        param_grid  = param_grid,
        lags_grid   = lags_grid,
        metric      = 'mean_absolute_error',
        return_best = True,
        n_jobs      = 1,  # ← Sin procesamiento paralelo para que no genere error
        verbose     = False
    )
    

    resultados_grid.to_csv(DESTINO_METRICAS / f"grilla_{nombre_zona}", index=False, encoding='utf-8')



    # -----------------------------------------------------------------------------
    # Aplicación en cada zona de los hiperparámetros encontrados
    # -----------------------------------------------------------------------------

        
    print(resultados_grid["n_estimators"][0])
    n_estimators_ajustado = resultados_grid["n_estimators"][0]

    print(resultados_grid["max_depth"][0])
    max_depth_ajustado = resultados_grid["max_depth"][0]

    print(resultados_grid["lags"][0])
    lags_array = resultados_grid["lags"][0]
    ventana_ajustada = lags_array[-1]
    ventana_ajustada2 = int(ventana_ajustada)

    forecaster_ajustado = ForecasterRecursive(
        regressor=RandomForestRegressor(
            n_estimators=n_estimators_ajustado,
            max_depth=max_depth_ajustado,
            random_state=123
        ),
        lags=ventana_ajustada2
    )
    
    forecaster_ajustado.fit(
        y=df_train['USAGE_KB'],  # target
        exog=df_train[exog_variables]    # Includes exogenous variables
    )

    predictions_ajustado = forecaster_ajustado.predict(
        steps=steps,
        exog=df_test[exog_variables]  # Future exogenous variables
    )



    # ------------------------------------------------------------------------------
    # Calculo de errores de la predicción hecha con los hiperparámetros encontrados:
    # ------------------------------------------------------------------------------


    # Mean Absolute Percentage Error
    mape_ajustado = mean_absolute_percentage_error(
        y_true=df_test['USAGE_KB'],
        y_pred=predictions_ajustado
    )
    mape_ajustado = round(mape_ajustado, 6)
    mape_percent_ajustado = mape_ajustado*100
    mape_percent_ajustado = round(mape_percent_ajustado, 4)
    print(f"MAPE: {mape_ajustado:.4f} ({mape_ajustado*100:.2f}%)")

    # Mean Absolute Error
    mae_ajustado = mean_absolute_error(df_test['USAGE_KB'], predictions_ajustado)
    mae_ajustado = round(mae_ajustado, 3)
    print(f"MAE: {mae_ajustado:.2f}")

    # Root Mean Squared Error
    mse_ajustado = mean_squared_error(df_test['USAGE_KB'], predictions_ajustado)
    rmse_ajustado = np.sqrt(mse_ajustado)
    rmse_ajustado = round(rmse_ajustado)
    print(f"RMSE: {rmse_ajustado:.2f}")

    r2_ajustado = r2_score(df_test['USAGE_KB'], predictions_ajustado)
    r2_ajustado = round(r2_ajustado, 3)
    print(f"R-Cuadrado: {r2_ajustado:.4f}")

    #new_row = pd.DataFrame([{"Zona": nombre_zona, "Tecnica": "Con Hiperparametros", "MAPE": mape, "MAPE(%)": mape_percent, "MAE": mae, "RMSE": rmse, "R2": r2}])
    new_row = pd.DataFrame([{"Zona": nombre_zona, "Modelo": "Random Forest", "MAPE_Base": mape_base, "MAPE_Optimizado": mape_ajustado, "MAPE(%)_Base": mape_percent_base, "MAPE(%)_Optimizado": mape_percent_ajustado, "MAE_Base": mae_base, "MAE_Optimizado": mae_ajustado, "RMSE_Base": rmse_base, "RMSE_Optimizado": rmse_ajustado, "R2_Base": r2_base, "R2_Optimizado": r2_ajustado}])
    df_errors = pd.concat([df_errors, new_row], ignore_index=True)
    df_errors.to_csv(DESTINO_METRICAS / "metricas_horizontales.csv", index=False, encoding='utf-8')
    

    usage_kb_compared = pd.DataFrame({
        'USAGE_KB_predicho': predictions_ajustado,
        'USAGE_KB_real': df_test['USAGE_KB']
    })
    difference = predictions_ajustado - df_test['USAGE_KB']
    usage_kb_compared['error_absoluto'] = difference.abs()
    usage_kb_compared['error_relativo'] = usage_kb_compared['error_absoluto'] / usage_kb_compared['USAGE_KB_real']

    usage_kb_compared.to_csv(DESTINO_METRICAS / f"metricas_hiperparametros_{nombre_zona}", index=False, encoding='utf-8')


    
    # -----------------------------------------------------------------------------
    # Graficacion serie temporal con la prediccion comparada con el test
    # -----------------------------------------------------------------------------


    #print(plt.style.available)
    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(25, 4))
    plt.plot(df_train['USAGE_KB'], label="Train", linewidth=2)
    plt.plot(df_test['USAGE_KB'], label="Test", linewidth=2)
    plt.plot(predictions_ajustado, label="Predicho", linewidth=2)
    plt.title(f"Random Forest - {nombre_zona}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, f"{nombre_zona}_serie.png"), dpi=300)
    plt.close()


    # ------------------------------------------------------------------------
    # Guardado del modelo para cada zona en archivos .pkl
    # ------------------------------------------------------------------------

    forecaster_futuro = ForecasterRecursive(
        regressor=RandomForestRegressor(
            n_estimators=n_estimators_ajustado,
            max_depth=max_depth_ajustado,
            random_state=123
        ),
        lags=10
    )
    
    forecaster_futuro.fit(
        y=df['USAGE_KB'],
        exog=df[exog_variables]
    )

    modelo_completo = {
        'forecaster': forecaster_futuro,
        'variables_config': {
            'exog_variables': exog_variables,          # ['DIA_SEMANA', ...]
            'target_column': 'USAGE_KB'
        },
        'metadata': {
            'zona': nombre_zona_recortado,
            'fecha_entrenamiento': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Random Forest',
            'kernel': 'rbf',
            'n_estimators': n_estimators_ajustado, 
            'max_depth': max_depth_ajustado, 
            'random_state': 123,
            'lags': 10
        }
    }

    # Guardar con joblib (mejor que pickle para objetos scikit-learn)
    joblib.dump(modelo_completo, MODELOS_DIR / f"RandomForest_{nombre_zona_recortado}.joblib")

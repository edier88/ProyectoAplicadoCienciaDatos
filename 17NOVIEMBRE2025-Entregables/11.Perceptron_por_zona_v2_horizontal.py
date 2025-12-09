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
from sklearn.neural_network import MLPRegressor
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
from pathlib import Path

GRAF_DIR = "Perceptron_Graficas"
os.makedirs(GRAF_DIR, exist_ok=True)

ORIGEN = "csv-zonas-wifi-separados-man/"
#DESTINO_METRICAS = "Random_Forest_Metricas"
DESTINO_METRICAS = Path("Perceptron_Metricas")
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
    print(f"\nProcesando: {nombre_zona}")
    
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
    steps = rows*0.3
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

    # Perceptron MLP
    forecaster = ForecasterRecursive(
        regressor=MLPRegressor(random_state=123),
        lags=10
    )

    # Ajuste con variables exogenas
    forecaster.fit(
        y=df_train['USAGE_KB_scaled'],
        exog=df_train[exog_variables_scaled]  # Add exogenous variables here
    )

    # Make predictions (returns scaled predictions)
    predictions_scaled = forecaster.predict(
        steps=steps,
        exog=df_test[exog_variables_scaled]  # Future scaled exogenous variables
    )

    # Descaling the predictions
    predictions_final = pd.Series(
        scaler_usage.inverse_transform(predictions_scaled.values.reshape(-1, 1)).flatten(),
        index=predictions_scaled.index
    )

    # Mean Absolute Percentage Error
    mape_base = mean_absolute_percentage_error(
        y_true=df_test['USAGE_KB'],
        y_pred=predictions_final
    )
    mape_percent_base = mape_base*100
    mape_base = round(mape_base, 3)
    mape_percent_base = round(mape_percent_base, 3)
    print(f"MAPE: {mape_base:.4f} ({mape_base*100:.2f}%)")

    # Mean Absolute Error
    mae_base = mean_absolute_error(df_test['USAGE_KB'], predictions_final)
    mae_base = round(mae_base, 3)
    print(f"MAE: {mae_base:.2f}")

    # Root Mean Squared Error
    mse_base = mean_squared_error(df_test['USAGE_KB'], predictions_final)
    rmse_base = np.sqrt(mse_base)
    rmse_base = round(rmse_base, 3)
    print(f"RMSE: {rmse_base:.2f}")

    r2_base = r2_score(df_test['USAGE_KB'], predictions_final)
    r2_base = round(r2_base, 3)
    print(f"R-Cuadrado: {r2_base:.2f}")

    # Construcción de CSV con métricas de errores de la predicción
    #new_row = pd.DataFrame([{"Zona": nombre_zona, "Tecnica": "Sin Hiperparametros", "MAPE": mape, "MAPE(%)": mape_percent, "MAE": mae, "RMSE": rmse, "R2": r2}])
    #df_errors = pd.concat([df_errors, new_row], ignore_index=True)
    #df_errors.to_csv(DESTINO_METRICAS / "metricas.csv", index=False, encoding='utf-8')

    #print(predictions,df_test['USAGE.KB'])
    usage_kb_compared = pd.DataFrame({
        'USAGE_KB_predicho': predictions_final,
        'USAGE_KB_real': df_test['USAGE_KB']
    })
    difference = predictions_final - df_test['USAGE_KB']
    usage_kb_compared['error_absoluto'] = difference.abs()
    usage_kb_compared['error_relativo'] = usage_kb_compared['error_absoluto'] / usage_kb_compared['USAGE_KB_real']
    #print(usage_kb_compared)

    usage_kb_compared.to_csv(DESTINO_METRICAS / f"metricas_{nombre_zona}", index=False, encoding='utf-8')
    
    # Graficar serie temporal
    #print(plt.style.available)
    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(10, 5))
    plt.plot(df_train['USAGE_KB'], label="Train", linewidth=2)
    plt.plot(df_test['USAGE_KB'], label="Test", linewidth=2)
    plt.plot(predictions_final, label="Predicho", linewidth=2)
    plt.title(f"Random Forest - {nombre_zona}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, f"{nombre_zona}_serie.png"), dpi=300)
    plt.close()



    # --------------------------------------------------------
    # Busqueda de hiperparámetros por zona
    # --------------------------------------------------------
    
    forecaster = ForecasterRecursive(
        regressor=MLPRegressor(random_state=123, max_iter=10000),
        lags=10
    )
    
    # MLP hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (50, 25, 10)],
        'activation': ['relu', 'tanh'], # relu (rectifier linear unit), tanh ()
        'solver': ['adam'], # adam, robusto a los picos
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01]
    }
    
    # Cross-validation
    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=int(len(df_train) * 0.8),
        refit=False,
        fixed_train_size=False
    )
    
    # Grid search
    resultados_grid = grid_search_forecaster(
        forecaster=forecaster,
        y=df_train['USAGE_KB_scaled'],
        exog=df_train[exog_variables_scaled],
        cv=cv,
        param_grid=param_grid,
        lags_grid=[10],  # Also optimize lags
        metric='mean_absolute_percentage_error',
        return_best=False,  # Keep all results to see everything
        n_jobs=1,  # MLP doesn't parallelize well, use 1
        verbose=False
    )

    resultados_grid.to_csv(DESTINO_METRICAS / f"grilla_{nombre_zona}", index=False, encoding='utf-8')



    # -----------------------------------------------------------------------------
    # Aplicación en cada zona de los hiperparámetros encontrados
    # -----------------------------------------------------------------------------
    
    print("lags:")
    print(resultados_grid["lags"][0])
    print("params:")
    print(resultados_grid["params"][0])
    print("mape:")
    print(resultados_grid["mean_absolute_percentage_error"][0])
    print("activation:")
    print(resultados_grid["activation"][0])
    print("alpha:")
    print(resultados_grid["alpha"][0])
    print("hidden_layer_sizes:")
    print(resultados_grid["hidden_layer_sizes"][0])
    print("learning_rate:")
    print(resultados_grid["learning_rate"][0])
    print("learning_rate_init:")
    print(resultados_grid["learning_rate_init"][0])
    print("solver:")
    print(resultados_grid["solver"][0])


    
    print(resultados_grid["lags"][0])
    print(type(resultados_grid["lags"][0]))
    lags_array = resultados_grid["lags"][0]
    print("ultimo lag:")
    print(lags_array[-1])

    ventana_ajustada = lags_array[-1]
    print("tipo de dato ventana_ajustada:")
    print(type(ventana_ajustada))
    ventana_ajustada2 = int(ventana_ajustada)
    print(type(ventana_ajustada2))

    hidden_layer_ajustado = resultados_grid["hidden_layer_sizes"][0]
    activation_ajustado = resultados_grid["activation"][0]
    solver_ajustado = resultados_grid["solver"][0]
    alpha_ajustado = resultados_grid["alpha"][0]
    learning_rate_ajustado = resultados_grid["learning_rate"][0]
    learning_rate_init_ajustado = resultados_grid["learning_rate_init"][0]


    
    # Nueva prediccion basado en los hiperparámetros encontrados de la grilla de cada zona:
    forecaster_ajustado = ForecasterRecursive(
        regressor=MLPRegressor(
            hidden_layer_sizes=hidden_layer_ajustado,
            activation=activation_ajustado,
            solver=solver_ajustado,
            max_iter=10000,
            random_state=123,
            alpha=alpha_ajustado,
            learning_rate=learning_rate_ajustado,
            learning_rate_init=learning_rate_init_ajustado
        ),
        lags=ventana_ajustada2
    )

    forecaster_ajustado.fit(
        y=df_train['USAGE_KB_scaled'],  # Scaled target
        exog=df_train[exog_variables_scaled]    # Includes scaled exogenous variables
    )

    predictions_scaled_ajustado = forecaster_ajustado.predict(
        steps=steps,
        exog=df_test[exog_variables_scaled]  # Future scaled exogenous variables
    )

    # Desescalado de la prediccion con los hiperparámetros encontrados en la grilla:
    predictions_final_ajustado = pd.Series(
        scaler_usage.inverse_transform(predictions_scaled_ajustado.values.reshape(-1, 1)).flatten(),
        index=predictions_scaled_ajustado.index
    )

    

    # ------------------------------------------------------------------------------
    # Calculo de errores de la predicción hecha con los hiperparámetros encontrados:
    # ------------------------------------------------------------------------------
    

    # Mean Absolute Percentage Error
    mape_ajustado = mean_absolute_percentage_error(
        y_true=df_test['USAGE_KB'],
        y_pred=predictions_final_ajustado
    )
    mape_percent_ajustado = mape_ajustado*100
    mape_ajustado = round(mape_ajustado, 3)
    mape_percent_ajustado = round(mape_percent_ajustado, 3)
    print(f"MAPE: {mape_ajustado:.4f} ({mape_ajustado*100:.2f}%)")

    # Mean Absolute Error
    mae_ajustado = mean_absolute_error(df_test['USAGE_KB'], predictions_final_ajustado)
    mae_ajustado = round(mae_ajustado, 3)
    print(f"MAE: {mae_ajustado:.2f}")

    # Root Mean Squared Error
    mse_ajustado = mean_squared_error(df_test['USAGE_KB'], predictions_final_ajustado)
    rmse_ajustado = np.sqrt(mse_ajustado)
    rmse_ajustado = round(rmse_ajustado, 3)
    print(f"RMSE: {rmse_ajustado:.2f}")

    r2_ajustado = r2_score(df_test['USAGE_KB'], predictions_final_ajustado)
    r2_ajustado = round(r2_ajustado, 3)
    print(f"R-Cuadrado: {r2_ajustado:.4f}")

    new_row = pd.DataFrame([{"Zona": nombre_zona, "Modelo": "Perceptron", "MAPE_Base": mape_base, "MAPE_Optimizado": mape_ajustado, "MAPE(%)_Base": mape_percent_base, "MAPE(%)_Optimizado": mape_percent_ajustado, "MAE_Base": mae_base, "MAE_Optimizado": mae_ajustado, "RMSE_Base": rmse_base, "RMSE_Optimizado": rmse_ajustado, "R2_Base": r2_base, "R2_Optimizado": r2_ajustado}])
    df_errors = pd.concat([df_errors, new_row], ignore_index=True)
    df_errors.to_csv(DESTINO_METRICAS / "metricas_horizontales.csv", index=False, encoding='utf-8')

    usage_kb_compared = pd.DataFrame({
        'USAGE_KB_predicho': predictions_final_ajustado,
        'USAGE_KB_real': df_test['USAGE_KB']
    })
    difference = predictions_final_ajustado - df_test['USAGE_KB']
    usage_kb_compared['error_absoluto'] = difference.abs()
    usage_kb_compared['error_relativo'] = usage_kb_compared['error_absoluto'] / usage_kb_compared['USAGE_KB_real']

    usage_kb_compared.to_csv(DESTINO_METRICAS / f"metricas_hiperparametros_{nombre_zona}", index=False, encoding='utf-8')
    

    # Display results
    #results_df = pd.DataFrame(resultados_grid)
    #results_df = results_df.sort_values('mean_squared_error')

#df_errors.to_csv(DESTINO_METRICAS / "metricas.csv", index=False, encoding='utf-8')
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

GRAF_DIR = "SVR_Graficas_TrainTest"
os.makedirs(GRAF_DIR, exist_ok=True)

MODELOS_DIR = Path("SVR_Modelos_Guardados")
os.makedirs(MODELOS_DIR, exist_ok=True)

ORIGEN = "csv-zonas-wifi-separados-man/"
#DESTINO_METRICAS = "Random_Forest_Metricas"
DESTINO_METRICAS = Path("SVR_Metricas")
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
    df_train, df_test = dividir_train_test(df_con_lags, test_size=0.3)

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

     # Inicializar escaladores para cada variable
    scaler_usage = RobustScaler()
    scaler_conexiones = RobustScaler()
    scaler_porcentaje = RobustScaler()

    # 1. Identificar las columnas que empiezan con el prefijo
    USAGE_KB_to_scale = df_train.filter(like='USAGE_KB').columns
    NUMERO_CONEXIONES_to_scale = df_train.filter(like='NUMERO_CONEXIONES').columns
    PORCENTAJE_USO_to_scale = df_train.filter(like='PORCENTAJE_USO').columns

    # Se escala la data de training
    df_train[USAGE_KB_to_scale] = scaler_usage.fit_transform(df_train[USAGE_KB_to_scale])
    df_train[NUMERO_CONEXIONES_to_scale] = scaler_conexiones.fit_transform(df_train[NUMERO_CONEXIONES_to_scale])
    df_train[PORCENTAJE_USO_to_scale] = scaler_porcentaje.fit_transform(df_train[PORCENTAJE_USO_to_scale])

    # Se escala test aplicando los escaladores de train (Se usa "transform", no "fit_transform")
    df_test[USAGE_KB_to_scale] = scaler_usage.transform(df_test[USAGE_KB_to_scale])
    df_test[NUMERO_CONEXIONES_to_scale] = scaler_conexiones.transform(df_test[NUMERO_CONEXIONES_to_scale])
    df_test[PORCENTAJE_USO_to_scale] = scaler_porcentaje.transform(df_test[PORCENTAJE_USO_to_scale])

    # Se escala df_con_lags aplicando los escaladores de train (Se usa "transform", no "fit_transform", este último se debe usar en la data a predecir)
    df_con_lags[USAGE_KB_to_scale] = scaler_usage.fit_transform(df_con_lags[USAGE_KB_to_scale])
    df_con_lags[NUMERO_CONEXIONES_to_scale] = scaler_conexiones.fit_transform(df_con_lags[NUMERO_CONEXIONES_to_scale])
    df_con_lags[PORCENTAJE_USO_to_scale] = scaler_porcentaje.fit_transform(df_con_lags[PORCENTAJE_USO_to_scale])


    # Guardado de CSV's de Train y Test ventaneados
    #df_train.to_csv(DESTINO_CSV_VENTANEADO / f"{nombre_zona_recortado}_train_v4.csv", index=False, encoding='utf-8')
    #df_test.to_csv(DESTINO_CSV_VENTANEADO / f"{nombre_zona_recortado}_test_v4.csv", index=False, encoding='utf-8')

    # Se crea y se entrena forecaster con SVR
    forecaster = ForecasterRecursive(
        regressor=SVR(),  # Using SVR instead of RandomForest
        lags=10
    )

    # Se usan los datos escalados
    forecaster.fit(
        y=df_train['USAGE_KB'],  # Scaled target
        exog=df_train[todas_variables_entrada]    # Includes scaled exogenous variables
    )

    steps = len(df_test)

    # Se hace la predicción escalada
    predictions_scaled = forecaster.predict(
        steps=steps,
        exog=df_test[todas_variables_entrada]
    )

    

    # --------------------------------------------------------
    # Busqueda de hiperparámetros por zona
    # --------------------------------------------------------


    forecaster_svr = ForecasterRecursive(
        regressor=SVR(kernel='rbf'),
        lags=10 # será reemplazado por el lag encontrado de la grilla
    )
    # Define cross-validation
    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=int(len(df_train) * 0.8),  # Use 60% for initial training
        refit=False,
        fixed_train_size=False
    )
    
    # SVR-specific hyperparameter grid
    param_grid = {
        'kernel': ['rbf'],
        'C': [0.1, 1.0, 10.0, 50, 100],
        'epsilon': [0.1, 0.01, 0.004],
        'gamma': ['scale', 'auto', 0.5, 0.05, 0.005]
    }

    # Busqueda de hiperparámetros por zona
    resultados_grid_svr = grid_search_forecaster(
        forecaster=forecaster_svr,
        y=df_train['USAGE_KB'], # el train sigue escalado, así SVR busca los mejores hiperparámetros con datos escalados
        exog=df_train[todas_variables_entrada],
        param_grid=param_grid,
        cv=cv,
        lags_grid=[10],
        metric='mean_absolute_error',
        return_best=True,
        n_jobs=1,  # ← Sin procesamiento paralelo para que no genere error
        verbose=False
    )

    resultados_grid_svr.to_csv(DESTINO_METRICAS / f"grilla_v4_{nombre_zona}", index=False, encoding='utf-8')


    # -----------------------------------------------------------------------------
    # Aplicación en cada zona de los hiperparámetros encontrados
    # -----------------------------------------------------------------------------


    print(resultados_grid_svr["lags"][0])
    print(type(resultados_grid_svr["lags"][0]))
    print(resultados_grid_svr["C"][0])
    print(resultados_grid_svr["epsilon"][0])
    print(resultados_grid_svr["gamma"][0])
    lags_array = resultados_grid_svr["lags"][0]
    print("ultimo lag:")
    print(lags_array[-1])

    ventana_ajustada = lags_array[-1]
    print("tipo de dato ventana_ajustada:")
    print(type(ventana_ajustada))
    ventana_ajustada2 = int(ventana_ajustada)
    print(type(ventana_ajustada2))
    C_ajustado = resultados_grid_svr["C"][0]
    epsilon_ajustado = resultados_grid_svr["epsilon"][0]
    gamma_ajustado = resultados_grid_svr["gamma"][0]

    
    # Nueva prediccion basado en los hiperparámetros encontrados de la grilla de cada zona:
    forecaster_ajustado = ForecasterRecursive(
        regressor=SVR(
            kernel='rbf', 
            C=C_ajustado, 
            epsilon=epsilon_ajustado, 
            gamma=gamma_ajustado
        ),
        lags=ventana_ajustada2
    )

    forecaster_ajustado.fit(
        y=df_train['USAGE_KB'],  # Scaled target
        exog=df_train[todas_variables_entrada]    # Includes scaled exogenous variables
    )

    predictions_scaled_ajustado = forecaster_ajustado.predict(
        steps=steps,
        exog=df_test[todas_variables_entrada]  # Future scaled exogenous variables
    )


    # ------------------------------------------------------------------------------
    # Se desescala el target del test para preparar los datos para la comparación con las predicciones
    # ------------------------------------------------------------------------------

    df_test['USAGE_KB'] = pd.Series(
        scaler_usage.inverse_transform(df_test['USAGE_KB'].values.reshape(-1, 1)).flatten(),
        index=df_test['USAGE_KB'].index
    )


    # ------------------------------------------------------------------------------
    # Calculo de errores de la predicción hecha SIN hiperparámetros:
    # ------------------------------------------------------------------------------

    # Desescalamiento de las predicciones
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
    print(f"R-Cuadrado: {r2_base:.4f}")

    # Errores absolutos y relativos de las predicciones desescaladas
    usage_kb_compared = pd.DataFrame({
        'USAGE_KB_predicho': predictions_final,
        'USAGE_KB_real': df_test['USAGE_KB']
    })
    difference = predictions_final - df_test['USAGE_KB']
    usage_kb_compared['error_absoluto'] = difference.abs()
    usage_kb_compared['error_relativo'] = usage_kb_compared['error_absoluto'] / usage_kb_compared['USAGE_KB_real']

    # Se guardan los errores absolutos y relativos en un CSV
    usage_kb_compared.to_csv(DESTINO_METRICAS / f"metricas_v4_{nombre_zona}", index=False, encoding='utf-8')


    # ------------------------------------------------------------------------------
    # Calculo de errores de la predicción hecha con los hiperparámetros encontrados:
    # ------------------------------------------------------------------------------

    # Desescalado de la prediccion con los hiperparámetros encontrados en la grilla:
    predictions_final_ajustado = pd.Series(
        scaler_usage.inverse_transform(predictions_scaled_ajustado.values.reshape(-1, 1)).flatten(),
        index=predictions_scaled_ajustado.index
    )

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

    new_row = pd.DataFrame([{"Zona": nombre_zona, "Modelo": "SVR", "MAPE_Base": mape_base, "MAPE_Optimizado": mape_ajustado, "MAPE(%)_Base": mape_percent_base, "MAPE(%)_Optimizado": mape_percent_ajustado, "MAE_Base": mae_base, "MAE_Optimizado": mae_ajustado, "RMSE_Base": rmse_base, "RMSE_Optimizado": rmse_ajustado, "R2_Base": r2_base, "R2_Optimizado": r2_ajustado}])
    df_errors = pd.concat([df_errors, new_row], ignore_index=True)
    df_errors.to_csv(DESTINO_METRICAS / "metricas_horizontales_v4.csv", index=False, encoding='utf-8')

    usage_kb_compared = pd.DataFrame({
        'USAGE_KB_predicho': predictions_final_ajustado,
        'USAGE_KB_real': df_test['USAGE_KB']
    })
    difference = predictions_final_ajustado - df_test['USAGE_KB']
    usage_kb_compared['error_absoluto'] = difference.abs()
    usage_kb_compared['error_relativo'] = usage_kb_compared['error_absoluto'] / usage_kb_compared['USAGE_KB_real']

    usage_kb_compared.to_csv(DESTINO_METRICAS / f"metricas_hiperparametros_v4_{nombre_zona}", index=False, encoding='utf-8')


    # ------------------------------------------------------------------------------
    # Se desescala el train para graficar la comparación con la predicción y el test
    # ------------------------------------------------------------------------------

    df_train['USAGE_KB'] = pd.Series(
        scaler_usage.inverse_transform(df_train['USAGE_KB'].values.reshape(-1, 1)).flatten(),
        index=df_train['USAGE_KB'].index
    )



    # ------------------------------------------------------------------------------
    # Graficación serie temporal para comparar test y predicho
    # ------------------------------------------------------------------------------

    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(25, 4))
    plt.plot(df_train['USAGE_KB'], label="Train", linewidth=2)
    plt.plot(df_test['USAGE_KB'], label="Test", linewidth=2)

    # Si el MAPE del modelo con hiperparámetros es menor al MAPE del modelo base, graficamos la predicción del modelo con los mejores hiperparámetros encontrados
    if mape_ajustado <= mape_base: 
        plt.plot(predictions_final_ajustado, label="Predicho", linewidth=2)
        plt.title(f"SVR Optimizado - {nombre_zona}")

    # Si el MAPE del modelo base es menor al MAPE del modelo con hiperparámetros, graficacmos la predicción del modelo base
    else:
        plt.plot(predictions_final, label="Predicho", linewidth=2)
        plt.title(f"SVR Base - {nombre_zona}")
    
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, f"{nombre_zona}_serie_v4.png"), dpi=300)
    plt.close()


    # ------------------------------------------------------------------------
    # Obtención de los últimos 10 días (ventana de diez días) del dataset original para guardarlo en el modelo
    # ------------------------------------------------------------------------

    # 1. Obtener los últimos días necesarios del histórico
    ultima_fecha_ventana = df.index[-1]
    
    # Necesitamos los últimos 'lags_exogenos' días para los lags
    # + 1 día extra para algunas variables
    dias_necesarios_ventana_datos = lags_exogenos + 1
    
    fecha_inicio_ventana = ultima_fecha_ventana - pd.Timedelta(days=dias_necesarios_ventana_datos - 1)
    
    # Filtrar la ventana
    ventana_datos = df.loc[fecha_inicio_ventana:ultima_fecha_ventana].copy()

    print("PILAS ESTA ES LA VENTA DE DATOS:")
    print(ventana_datos)


    #df['USAGE_KB_scaled'] = scaler_usage.fit_transform(df[['USAGE_KB']])
    #df['NUMERO_CONEXIONES_scaled'] = scaler_conexiones.fit_transform(df[['NUMERO_CONEXIONES']])
    #df['PORCENTAJE_USO_scaled'] = scaler_porcentaje.fit_transform(df[['PORCENTAJE_USO']])


    # ------------------------------------------------------------------------
    # Guardado del modelo para cada zona en archivos .joblib
    # ------------------------------------------------------------------------

    # Si el MAPE del modelo con hiperparámetros es menor al MAPE del modelo base, guardamos el modelo con los mejores hiperparámetros encontrados
    if mape_ajustado <= mape_base: 

        forecaster_a_guardar = ForecasterRecursive(
            regressor=SVR(
                kernel='rbf', 
                C=C_ajustado, 
                epsilon=epsilon_ajustado, 
                gamma=gamma_ajustado
            ),
            lags=ventana_ajustada2
        )

        tipo_modelo_guardado = "Optimizado"
        
    # Si el MAPE del modelo base es menor al MAPE del modelo con hiperparámetros, guardamos el modelo base
    else:
        
        forecaster_a_guardar = ForecasterRecursive(
            regressor=SVR(),  # Using SVR instead of RandomForest
            lags=10
        )

        tipo_modelo_guardado = "Base"
    
    # Entrenamos el modelo con las variables de entrada (exógenas) y de salida (target: USAGE_KB)
    forecaster_a_guardar.fit(
        y=df_con_lags['USAGE_KB'],  # Scaled target
        exog=df_con_lags[todas_variables_entrada]    # Includes scaled exogenous variables
    )


    # 5. Guardar TODO en un solo archivo
    modelo_completo = {
        'forecaster': forecaster_a_guardar,
        'ventana_datos': ventana_datos,
        'scalers': {
            'scaler_usage': scaler_usage,          # Para USAGE_KB
            'scaler_conexiones': scaler_conexiones,  # Para NUMERO_CONEXIONES
            'scaler_porcentaje': scaler_porcentaje   # Para PORCENTAJE_USO
        },
        'variables_config': {
            'exog_variables_original': exog_variables,          # ['DIA_SEMANA', ...]
            'exog_variables_scaled': todas_variables_entrada,     # ['DIA_SEMANA', ..., '_scaled']
            'target_column': 'USAGE_KB',
            'scaled_target_column': 'USAGE_KB'
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
            'tipo_modelo_guardado': tipo_modelo_guardado,
            'MAPE_Optimizado': mape_ajustado,
            'MAPE_Base': mape_base,
            'R2_Optimizado': r2_ajustado,
            'R2_Base': r2_base,
            'zona': nombre_zona_recortado,
            'fecha_entrenamiento': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'SVR',
            'kernel': 'rbf',
            'C_optimizado': C_ajustado, 
            'epsilon_optimizado': epsilon_ajustado, 
            'gamma_optimizado': gamma_ajustado,
            'lags': 10
        }
    }

    # Guardar con joblib (mejor que pickle para objetos scikit-learn)
    joblib.dump(modelo_completo, MODELOS_DIR / f"SVR_{nombre_zona_recortado}_v4.joblib")
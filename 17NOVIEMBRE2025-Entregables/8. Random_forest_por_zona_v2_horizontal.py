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
from pathlib import Path

GRAF_DIR = "Random_Forest_Graficas_TrainTest"
os.makedirs(GRAF_DIR, exist_ok=True)

GRAF_FUTURAS_DIR = "Random_Forest_Graficas_Futuras"
os.makedirs(GRAF_FUTURAS_DIR, exist_ok=True)

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


    # --------------------------------------------------
    # Creacion de variables de entrada (exógenas) para predicción real de una semana
    # --------------------------------------------------

    def crear_exog_futura(ultima_fecha, df_historico, steps=7):
        """
        Busca el mismo período del año anterior (7 días consecutivos).
        Si no encuentra, busca en meses siguientes.
        """
        
        # Crear rango de fechas futuras
        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Crear DataFrame para exógenas futuras
        exog_futura = pd.DataFrame(index=fechas_futuras)
        exog_futura['DIA_SEMANA'] = fechas_futuras.dayofweek
        exog_futura['LABORAL'] = (exog_futura['DIA_SEMANA'] < 5).astype(int)
        exog_futura['FIN_DE_SEMANA'] = (exog_futura['DIA_SEMANA'] >= 5).astype(int)
        exog_futura['FESTIVO'] = 0
        
        # BUSCAR PERIODO COMPLETO DEL AÑO ANTERIOR
        print(f"\nBuscando período del año anterior para {steps} días")
        
        # Calcular el período del año anterior
        inicio_periodo_anterior = fechas_futuras[0] - pd.DateOffset(years=1)
        fin_periodo_anterior = fechas_futuras[-1] - pd.DateOffset(years=1)
        
        print(f"Período buscado del año anterior: {inicio_periodo_anterior.date()} a {fin_periodo_anterior.date()}")
        
        # Intentar encontrar el período completo
        periodo_encontrado = False
        valores_conexiones = []
        valores_porcentaje = []
        
        # Estrategia 1: Buscar período exacto
        fechas_buscar = pd.date_range(start=inicio_periodo_anterior, periods=steps, freq='D')
        
        todas_fechas_encontradas = all(fecha in df_historico.index for fecha in fechas_buscar)
        
        if todas_fechas_encontradas:
            print("✓ Encontrado período exacto del año anterior")
            valores_conexiones = [df_historico.loc[fecha, 'NUMERO_CONEXIONES'] for fecha in fechas_buscar]
            valores_porcentaje = [df_historico.loc[fecha, 'PORCENTAJE_USO'] for fecha in fechas_buscar]
            periodo_encontrado = True
        
        # Estrategia 2: Buscar en el mismo mes del año anterior
        if not periodo_encontrado:
            print("⚠ No se encontró período exacto. Buscando en el mismo mes del año anterior...")
            
            mismo_mes = df_historico[
                (df_historico.index.month == inicio_periodo_anterior.month) & 
                (df_historico.index.year == inicio_periodo_anterior.year)
            ]
            
            if len(mismo_mes) >= steps:
                # Tomar los primeros 'steps' días del mes
                mismo_mes = mismo_mes.sort_index()
                valores_conexiones = mismo_mes['NUMERO_CONEXIONES'].head(steps).values
                valores_porcentaje = mismo_mes['PORCENTAJE_USO'].head(steps).values
                periodo_encontrado = True
                print(f"✓ Usando primeros {steps} días del mes {inicio_periodo_anterior.month}/{inicio_periodo_anterior.year}")
        
        # Estrategia 3: Buscar en meses siguientes
        if not periodo_encontrado:
            print("⚠ No hay suficientes datos en el mismo mes. Buscando en meses siguientes...")
            
            for offset_meses in range(1, 13):  # Buscar en los próximos 12 meses
                mes_buscar = inicio_periodo_anterior.month + offset_meses
                anio_buscar = inicio_periodo_anterior.year
                
                # Ajustar si se pasa de diciembre
                if mes_buscar > 12:
                    mes_buscar -= 12
                    anio_buscar += 1
                
                mes_siguiente = df_historico[
                    (df_historico.index.month == mes_buscar) & 
                    (df_historico.index.year == anio_buscar)
                ]
                
                if len(mes_siguiente) >= steps:
                    mes_siguiente = mes_siguiente.sort_index()
                    valores_conexiones = mes_siguiente['NUMERO_CONEXIONES'].head(steps).values
                    valores_porcentaje = mes_siguiente['PORCENTAJE_USO'].head(steps).values
                    periodo_encontrado = True
                    print(f"✓ Encontrado en mes {mes_buscar}/{anio_buscar} ({offset_meses} meses después)")
                    break
        
        # Estrategia 4: Usar promedio general como último recurso
        if not periodo_encontrado:
            print("⚠ No se encontró ningún período adecuado. Usando promedios generales...")
            valores_conexiones = [df_historico['NUMERO_CONEXIONES'].mean()] * steps
            valores_porcentaje = [df_historico['PORCENTAJE_USO'].mean()] * steps
        
        # Asignar valores
        exog_futura['NUMERO_CONEXIONES'] = valores_conexiones
        exog_futura['PORCENTAJE_USO'] = valores_porcentaje
        
        # Mostrar resumen
        print(f"\nValores asignados para los próximos {steps} días:")
        for i, fecha in enumerate(exog_futura.index):
            print(f"  {fecha.date()}: {valores_conexiones[i]:.1f} conexiones, {valores_porcentaje[i]:.1f}% uso")
        
        return exog_futura


    # Obtener última fecha de datos
    ultima_fecha = df.index[-1]
    print(f"\n Ultimo dia data frame original:")
    print(ultima_fecha)

    # Crear exógenas para 7 días futuros
    exog_7_dias = crear_exog_futura(ultima_fecha, df)

    print(f"\n Data frame futuro generado:")
    print(exog_7_dias)

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
    
    # Predecir 7 días con variables exógenas
    predicciones_7_dias = forecaster_futuro.predict(
        steps=7,
        exog=exog_7_dias[exog_variables]
    )

    #Guardado de las gráficas de predicciones futuras
    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(25, 4))
    plt.plot(df['USAGE_KB'], label="Tráfico Pasado", linewidth=2)
    plt.plot(predicciones_7_dias, label="Tráfico Predicho", linewidth=2)
    plt.title(f"Random Forest - {nombre_zona}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_FUTURAS_DIR, f"{nombre_zona}_prediccion_7dias.png"), dpi=300)
    plt.close()

    
#df_errors.to_csv(DESTINO_METRICAS / "metricas.csv", index=False, encoding='utf-8')
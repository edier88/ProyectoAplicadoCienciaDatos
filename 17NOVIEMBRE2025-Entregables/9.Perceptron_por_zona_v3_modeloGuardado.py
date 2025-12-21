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

MODELOS_DIR = Path("Perceptron_Modelos_Guardados")

ORIGEN = "csv-zonas-wifi-separados-man/"
#DESTINO_METRICAS = "Random_Forest_Metricas"
DESTINO_METRICAS = Path("Perceptron_Metricas")
os.makedirs(DESTINO_METRICAS, exist_ok=True)

GRAF_FUTURAS_DIR = "Perceptron_Graficas_Futuras"
os.makedirs(GRAF_FUTURAS_DIR, exist_ok=True)

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
    
    # Variables de entrada para el modelo
    exog_variables = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO', 'NUMERO_CONEXIONES']
    # Define las variables exogenas (usa las versiones escaladas para variables continuas)
    exog_variables_scaled = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO_scaled', 'NUMERO_CONEXIONES_scaled']

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

    modelo_completo = joblib.load(open(MODELOS_DIR / f"MLP_{nombre_zona_recortado}.joblib", 'rb'))

    forecaster = modelo_completo['forecaster']
    scaler_usage = modelo_completo['scalers']['scaler_usage']
    scaler_conexiones = modelo_completo['scalers']['scaler_conexiones']
    scaler_porcentaje = modelo_completo['scalers']['scaler_porcentaje']
    config = modelo_completo['variables_config']


    # Se escalan las exogenas futuras aplicando los escaladores de train (Se usa "transform", no "fit_transform")
    exog_7_dias['NUMERO_CONEXIONES_scaled'] = scaler_conexiones.transform(exog_7_dias[['NUMERO_CONEXIONES']])
    exog_7_dias['PORCENTAJE_USO_scaled'] = scaler_porcentaje.transform(exog_7_dias[['PORCENTAJE_USO']])

    predictions_scaled_futuro = forecaster.predict(
        steps=7,
        exog=exog_7_dias[exog_variables_scaled]  # Future scaled exogenous variables
    )

    # Desescalado de la prediccion con los hiperparámetros encontrados en la grilla:
    predictions_final_futuro = pd.Series(
        scaler_usage.inverse_transform(predictions_scaled_futuro.values.reshape(-1, 1)).flatten(),
        index=predictions_scaled_futuro.index
    )

    #Guardado de las gráficas de predicciones futuras
    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(25, 4))
    plt.plot(df['USAGE_KB'], label="Tráfico Pasado", linewidth=2)
    plt.plot(predictions_final_futuro, label="Tráfico Predicho", linewidth=2)
    plt.title(f"Perceptron - {nombre_zona}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Tráfico (kB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_FUTURAS_DIR, f"{nombre_zona}_prediccion_7dias.png"), dpi=300)
    plt.close()
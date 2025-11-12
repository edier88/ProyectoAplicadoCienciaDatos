
# solo_desescalado_y_validacion

"""
==============================================
SCRIPT: solo_desescalado_y_validacion.py
Autores: Equipo Tesis - Maestría en Ciencia de Datos
Fecha: 08/11/2025
----------------------------------------------
OBJETIVO GENERAL:
Este script realiza el desescalado de las predicciones generadas
por el modelo Random Forest (guardadas en la carpeta 
'predicciones_por_zona-con-R-21-10-2025/') y las compara con los
valores reales del conjunto de test (carpeta 'test-30/').

El propósito es evaluar el desempeño del modelo en unidades reales (KB),
obteniendo métricas como MAE, RMSE y Error Relativo (%).

----------------------------------------------
FLUJO LÓGICO DEL PROCESO
----------------------------------------------

1️⃣ CONFIGURACIÓN DE RUTAS
   Se definen las carpetas de entrada y salida:
   - predicciones_por_zona-con-R-21-10-2025/: resultados del modelo escalado
   - scalers_original/: contiene los archivos .pkl usados para escalar los datos
   - test-30/: datos reales sin escalar
   - results_descaled-08112025/: carpeta donde se guardarán los resultados finales

2️⃣ CARGA DE PREDICCIONES
   Se leen los .csv de predicciones de cada zona (una por archivo).

3️⃣ CARGA Y USO DEL SCALER CORRESPONDIENTE
   Se carga el scaler .pkl de la misma zona y se identifica la columna objetivo 
   (usage_kB o similar). Se aplica la transformación inversa de forma robusta 
   incluso si el scaler contiene más columnas de las usadas en predicción.

4️⃣ DESESCALADO MANUAL
   Se aplica la fórmula inversa X_real = X_scaled * scale + mean (o center_ si es RobustScaler)
   únicamente sobre las columnas y_true e y_pred.

5️⃣ COMPARACIÓN CON DATOS REALES
   Se leen los archivos reales del test-30/, se toma la columna USAGE.KB y se alinean
   las longitudes de ambas fuentes (pred vs real).

6️⃣ CÁLCULO DE MÉTRICAS
   - MAE (Error absoluto medio)
   - RMSE (Raíz del error cuadrático medio)
   - Error Relativo (%)

7️⃣ EXPORTACIÓN DE RESULTADOS
   Para cada zona se genera un archivo con columnas:
   USAGE_KB_real | USAGE_KB_pred | Error_abs_KB | Error_rel_%
   y un log global con todas las métricas por zona.

8️⃣ MANEJO DE ERRORES Y LOG
   Si una zona no tiene scaler o test asociado, se registra en el log con el estado correspondiente.

==============================================
"""

# ========= IMPORTACIONES =========
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ========= CONFIGURACIÓN =========
predicciones_folder = "predicciones_por_zona-con-R-21-10-2025/"
scalers_folder = "scalers_original/"
test_original_folder = "test-30/"
output_folder = "results_descaled-08112025/"
os.makedirs(output_folder, exist_ok=True)

print("\n=== Iniciando desescalado y validación ===\n")

log = []
csv_files = [f for f in os.listdir(predicciones_folder) if f.endswith(".csv")]

# ========= PROCESO PRINCIPAL =========
for file in csv_files:
    zona_name = file.replace("_pred.csv", "")
    print(f"Procesando zona: {zona_name}")

    # 1️⃣ Cargar predicciones escaladas
    pred_path = os.path.join(predicciones_folder, file)
    df_pred = pd.read_csv(pred_path)

    # 2️⃣ Cargar scaler correspondiente
    scaler_path = os.path.join(scalers_folder, f"{zona_name}.pkl")
    if not os.path.exists(scaler_path):
        print(f"⚠️ No se encontró scaler para {zona_name}")
        log.append([zona_name, "Sin scaler"])
        continue

    scaler = joblib.load(scaler_path)

    # 3️⃣ Desescalar las columnas y_true e y_pred
    try:
        features_scaler = list(getattr(scaler, "feature_names_in_", []))

        # Caso A: el scaler no tiene metadata -> usar método directo
        if not features_scaler:
            print(f"⚠️ Scaler sin metadatos, se aplicará desescalado simple...")
            df_pred['y_true_des'] = scaler.inverse_transform(df_pred[['y_true']])[:, 0]
            df_pred['y_pred_des'] = scaler.inverse_transform(df_pred[['y_pred']])[:, 0]

        else:
            # Caso B: el scaler tiene metadata -> identificar columna objetivo
            target_cols = [c for c in features_scaler if 'usage' in c.lower() or 'y' in c.lower()]
            if not target_cols:
                raise ValueError("No se encontró la columna de target en el scaler")

            target_index = features_scaler.index(target_cols[0])

            # Obtener escala y media del target (según tipo de scaler)
            if hasattr(scaler, "center_"):  # RobustScaler
                mean_ = scaler.center_[target_index]
            else:  # StandardScaler / MinMaxScaler
                mean_ = scaler.mean_[target_index]

            scale_ = scaler.scale_[target_index]

            # Aplicar fórmula inversa: X_real = X_scaled * scale + mean
            df_pred['y_true_des'] = df_pred['y_true'] * scale_ + mean_
            df_pred['y_pred_des'] = df_pred['y_pred'] * scale_ + mean_

    except Exception as e:
        print(f"❌ Error desescalando {zona_name}: {e}")
        log.append([zona_name, f"Error desescalado: {e}"])
        continue

    # 4️⃣ Cargar archivo original del test
    test_file = os.path.join(test_original_folder, f"{zona_name}.csv")
    if not os.path.exists(test_file):
        print(f"⚠️ No se encontró test original para {zona_name}")
        log.append([zona_name, "Sin test original"])
        continue

    df_test = pd.read_csv(test_file)

    # Buscar columna USAGE.KB (independiente de mayúsculas)
    col_usage = [c for c in df_test.columns if 'USAGE' in c.upper()]
    if not col_usage:
        print(f"⚠️ No se encontró columna USAGE.KB en {zona_name}")
        log.append([zona_name, "Sin columna USAGE.KB"])
        continue

    df_test_usage = df_test[col_usage[0]].reset_index(drop=True)

    # 5️⃣ Alinear longitudes
    n = min(len(df_pred), len(df_test_usage))
    df_pred = df_pred.head(n)
    df_test_usage = df_test_usage.head(n)

    # 6️⃣ Calcular métricas
    y_true_real = df_test_usage.values
    y_pred_real = df_pred['y_pred_des'].values

    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    err_rel = np.mean(np.abs(y_pred_real - y_true_real) / (y_true_real + 1e-9)) * 100

    # 7️⃣ Guardar resultados individuales
    df_result = pd.DataFrame({
        'USAGE_KB_real': y_true_real,
        'USAGE_KB_pred': y_pred_real,
        'Error_abs_KB': np.abs(y_pred_real - y_true_real),
        'Error_rel_%': np.abs(y_pred_real - y_true_real) / (y_true_real + 1e-9) * 100
    })

    result_path = os.path.join(output_folder, f"{zona_name}_desescalado_validado.csv")
    df_result.to_csv(result_path, index=False)

    log.append([zona_name, "✅ OK", mae, rmse, err_rel])
    print(f"✅ {zona_name}: MAE={mae:.1f} | RMSE={rmse:.1f} | ErrorRel={err_rel:.2f}%\n")

# ========= CORRECCIÓN PARA LOG FLEXIBLE =========
if log:
    max_cols = max(len(x) for x in log)
    for row in log:
        while len(row) < max_cols:
            row.append("")
    cols = ["Zona", "Estado", "MAE", "RMSE", "ErrorRel_%"][:max_cols]
    log_df = pd.DataFrame(log, columns=cols)
    log_df.to_csv(os.path.join(output_folder, "log_validacion.csv"), index=False)

print("=== Proceso finalizado ===")
print(f"Resumen global guardado en: {os.path.join(output_folder, 'log_validacion.csv')}\n")

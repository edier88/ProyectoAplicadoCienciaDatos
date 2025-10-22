#ejemplo-Regression-Parque-del-ingenio.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. CARGAR DATASETS
# Ajusta los nombres de archivo según cómo los tengan guardados
train_file = "train_windowed/agrupado_001_ZW Parque Ingenio.csv"
test_file  = "test_windowed/agrupado_001_ZW Parque Ingenio.csv"

df_train = pd.read_csv(train_file)
df_test  = pd.read_csv(test_file)

# 2. SEPARAR FEATURES Y TARGET
X_train = df_train[[f"lag{i}" for i in range(1, 8)]]
y_train = df_train["target"]

X_test = df_test[[f"lag{i}" for i in range(1, 8)]]
y_test = df_test["target"]

# 3. ENTRENAR MODELO
model = LinearRegression()
model.fit(X_train, y_train) # Entrenamiento del modelo

# 4. PREDICCIONES Y MÉTRICAS
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== Resultados Parque Ingenio ===")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)

# 5. GRÁFICO RESULTADOS
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Valores Reales", marker="o")
plt.plot(y_pred, label="Predicciones", marker="x")
plt.title("Parque Ingenio - Regresión Lineal (Predicción vs Real)")
plt.xlabel("Observaciones")
plt.ylabel("Tráfico KB (escalado)")
plt.legend()
plt.grid(True)
plt.show()

#simulacion-regresion-lineal.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Simulación de un dataset como el de ustedes (lag1...lag7 y target)
np.random.seed(42)
n = 200
data = pd.DataFrame({
    f"lag{i}": np.random.randn(n)*1000 for i in range(1, 8)
})
# target = combinación lineal de algunos lags + ruido
data["target"] = (0.3*data["lag1"] - 0.2*data["lag3"] + 0.5*data["lag7"] +
                  np.random.randn(n)*500)

# Separar features y target
X = data[[f"lag{i}" for i in range(1, 8)]]
y = data["target"]

# Split train (70%) / test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Crear y entrenar modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)

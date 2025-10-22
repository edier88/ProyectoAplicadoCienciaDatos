# ===========================================================
# plot_random_forest_parque_ingenio.py
# Genera dos gráficas para la zona ZW Parque Ingenio:
# 1. Serie temporal (y_true vs y_pred)
# 2. Dispersión (y_true vs y_pred)
# 07-10-2025
# ===========================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== CONFIGURACIÓN =====
CSV_PATH = "resultados_random_forest.csv"
ZONA_OBJETIVO = "ZW Parque Ingenio"   # cambia si el nombre difiere
OUTPUT_DIR = "graficas_random_forest"

# ===== CREAR DIRECTORIO DE SALIDA =====
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ===== 1. CARGAR DATOS =====
df = pd.read_csv(CSV_PATH)

# Verificar que exista la columna de zona
if "zona" not in df.columns:
    raise ValueError("❌ El archivo no contiene una columna llamada 'zona'. Agrega esa columna para identificar la zona WiFi.")

# Filtrar la zona de interés
df_zona = df[df["zona"].str.contains(ZONA_OBJETIVO, case=False, na=False)].copy()

if df_zona.empty:
    raise ValueError(f"❌ No se encontraron registros para la zona: {ZONA_OBJETIVO}")

# ===== 2. SERIE TEMPORAL =====
plt.figure(figsize=(10, 5))
plt.plot(df_zona.index, df_zona["y_true"], label="Valor real", linewidth=2)
plt.plot(df_zona.index, df_zona["y_pred"], label="Predicción (Random Forest)", linewidth=2, linestyle="--")
plt.title(f"Random Forest - {ZONA_OBJETIVO}")
plt.xlabel("Índice temporal (observaciones)")
plt.ylabel("Tráfico (kB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_path_serie = os.path.join(OUTPUT_DIR, f"{ZONA_OBJETIVO.replace(' ', '_')}_serie.png")
plt.savefig(output_path_serie, dpi=300)
plt.close()
print(f"✅ Gráfica temporal guardada en: {output_path_serie}")

# ===== 3. GRÁFICA DE DISPERSIÓN =====
plt.figure(figsize=(6, 6))
plt.scatter(df_zona["y_true"], df_zona["y_pred"], alpha=0.6, color="#007acc", label="Predicciones")
plt.plot(
    [df_zona["y_true"].min(), df_zona["y_true"].max()],
    [df_zona["y_true"].min(), df_zona["y_true"].max()],
    color="red", linestyle="--", linewidth=2, label="Línea perfecta (y = x)"
)
plt.xlabel("Valor real (y_true)")
plt.ylabel("Predicción (y_pred)")
plt.title(f"Dispersión Real vs Predicho - {ZONA_OBJETIVO}")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_path_disp = os.path.join(OUTPUT_DIR, f"{ZONA_OBJETIVO.replace(' ', '_')}_dispersion.png")
plt.savefig(output_path_disp, dpi=300)
plt.close()
print(f"✅ Gráfica de dispersión guardada en: {output_path_disp}")

print("\n🎨 Generación de gráficas completada exitosamente.")

# train_predict_global_zona.py

import pandas as pd
import numpy as np

# intenta usar LightGBM; si no está instalado, usa HistGradientBoostingRegressor (Scikit-Learn)
try:
    import lightgbm as lgb
    USE_LGB = True
except Exception:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_LGB = False

from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils_trafico import (
    ensure_datetime, add_time_features, build_zone_panel_with_imputation,
    add_lags_sorted, compute_ap_proportions, simple_walkforward_splits
)

# =================== 1) CARGA ===================
df = pd.read_csv("trafico_por_ap.csv")   # ajusta ruta
df = ensure_datetime(df, 'fecha')

# =================== 2) PANEL ZONA + IMPUTACIÓN ===================
zona = build_zone_panel_with_imputation(df)
zona = add_time_features(zona, 'fecha')
zona = add_lags_sorted(zona, group_col='zona', target='trafico_total_kB', lags=(1,7))

# quitar filas sin lags
zona_model = zona.dropna(subset=['lag1','lag7']).copy()
zona_model = zona_model.sort_values(['fecha','zona'])

# =================== 3) FEATURES ===================
features = ['zona','dow','is_weekend','conexiones_totales','lag1','lag7','flag_imputado_zona']
target = 'trafico_total_kB'

X = zona_model[features].copy()
y = zona_model[target].copy()

# codificar 'zona' como categórica o numérica
# LightGBM acepta categóricas si son dtype 'category'
X['zona'] = X['zona'].astype('category')

# =================== 4) VALIDACIÓN TEMPORAL (walk-forward simple, global) ===================
# orden por fecha global (todas las zonas mezcladas). Alternativa: validación por zona y promediar.
zona_model = zona_model.sort_values('fecha').reset_index(drop=True)
X = X.loc[zona_model.index]
y = y.loc[zona_model.index]

splits = simple_walkforward_splits(len(zona_model), n_splits=5, min_train=500)

maes, rmses = [], []
for train_idx, test_idx in splits:
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if USE_LGB:
        model = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        model.fit(X_train, y_train, categorical_feature=['zona','dow','is_weekend'])
    else:
        model = HistGradientBoostingRegressor(random_state=42)
        # one-hot simple para 'zona' si no usamos LightGBM
        X_train = pd.get_dummies(X_train, columns=['zona','dow','is_weekend'], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=['zona','dow','is_weekend'], drop_first=True)
        # alinear columnas
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        model.fit(X_train, y_train)

    if USE_LGB:
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    maes.append(mean_absolute_error(y_test, y_pred))
    rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))

print(f"MAE promedio (walk-forward): {np.mean(maes):.1f} kB")
print(f"RMSE promedio (walk-forward): {np.mean(rmses):.1f} kB")

# =================== 5) ENTRENAMIENTO FINAL ===================
if USE_LGB:
    final_model = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        n_estimators=800,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    final_model.fit(X, y, categorical_feature=['zona','dow','is_weekend'])
    encoder_info = None  # LightGBM maneja categóricas por dtype
else:
    final_model = HistGradientBoostingRegressor(random_state=42)
    # guardar columnas de entrenamiento tras dummies
    X_enc = pd.get_dummies(X, columns=['zona','dow','is_weekend'], drop_first=True)
    final_model.fit(X_enc, y)
    encoder_info = {'cols': X_enc.columns.tolist()}

# =================== 6) PREDICCIÓN H DÍAS FUTUROS POR ZONA ===================
H = 7  # horizonte (días)

# necesitamos proporciones históricas para reparto por AP
prop_hist = compute_ap_proportions(df)

# crear estructura para “simulación” autorregresiva por zona (usa lag1/lag7)
# último estado por zona
last_by_zone = (zona
                .sort_values(['zona','fecha'])
                .groupby('zona')
                .tail(7)  # necesitamos 7 días para lag7
                .copy())

# función para armar fila X de predicción por zona y fecha específica
def make_feature_row(zona_id, fecha_pred, history_df):
    # history_df: contiene últimos días de esa zona, con trafico_total_kB y conexiones_totales

    # lags
    # lag1 = último día observado
    last = history_df.sort_values('fecha').tail(1).iloc[0]
    lag1 = last['trafico_total_kB']

    # lag7: si hay 7 días, úsalo; si no, fallback = lag1
    if len(history_df) >= 7:
        lag7 = history_df.sort_values('fecha').iloc[-7]['trafico_total_kB']
    else:
        lag7 = lag1

    # conexiones: como punto de partida, usa el último valor observado (o media móvil si prefieres)
    conexiones = last['conexiones_totales']

    dow = fecha_pred.dayofweek
    is_weekend = int(dow in [5,6])

    row = pd.DataFrame([{
        'zona': zona_id,
        'dow': dow,
        'is_weekend': is_weekend,
        'conexiones_totales': conexiones,
        'lag1': lag1,
        'lag7': lag7,
        'flag_imputado_zona': 0  # durante predicción futura
    }])
    return row

# contenedor para predicciones
preds_all = []

# loop por zona
for zona_id, hist in zona.groupby('zona'):
    hist_zone = hist.copy()

    # simulación autoregresiva H pasos
    for h in range(1, H+1):
        fecha_pred = hist_zone['fecha'].max() + pd.Timedelta(days=1)

        x_row = make_feature_row(zona_id, fecha_pred, hist_zone)
        if USE_LGB:
            x_row['zona'] = x_row['zona'].astype('category')
            y_hat = final_model.predict(x_row)[0]
        else:
            x_row_enc = pd.get_dummies(x_row, columns=['zona','dow','is_weekend'], drop_first=True)
            # alinear columnas
            for col in encoder_info['cols']:
                if col not in x_row_enc.columns:
                    x_row_enc[col] = 0
            x_row_enc = x_row_enc[encoder_info['cols']]
            y_hat = final_model.predict(x_row_enc)[0]

        preds_all.append({'fecha': fecha_pred, 'zona': zona_id, 'trafico_total_kB_pred': y_hat})

        # actualizar "historia" para el siguiente paso (autorregresivo)
        hist_zone = pd.concat([
            hist_zone,
            pd.DataFrame([{
                'fecha': fecha_pred,
                'zona': zona_id,
                'trafico_total_kB': y_hat,
                'conexiones_totales': hist_zone['conexiones_totales'].iloc[-1]  # simple hold
            }])
        ], ignore_index=True)

pred_zona = pd.DataFrame(preds_all).sort_values(['zona','fecha']).reset_index(drop=True)

print("\nPredicción por zona (primeras filas):")
print(pred_zona.head())

# =================== 7) REPARTO POR AP (top-down) ===================
# unir DOW para cada fecha predicha
pred_zona['dow'] = pred_zona['fecha'].dt.dayofweek

# combinamos con prop_hist (zona, id_ap, dow) -> duplicamos predicción por cada AP de la zona
pred_por_ap = (pred_zona.merge(prop_hist, on=['zona','dow'], how='left')
                         .rename(columns={'prop_ap':'prop_ap_hist'}))

# fallback si no hay proporción histórica (p.ej., nuevo AP sin historia): 50/50
pred_por_ap['prop_ap_hist'] = pred_por_ap['prop_ap_hist'].fillna(0.5)

pred_por_ap['trafico_ap_pred_kB'] = pred_por_ap['trafico_total_kB_pred'] * pred_por_ap['prop_ap_hist']

# resultado final: tráfico predicho por AP
pred_por_ap = pred_por_ap[['fecha','zona','id_ap','trafico_ap_pred_kB','trafico_total_kB_pred']]

print("\nPredicción por AP (primeras filas):")
print(pred_por_ap.head())

# =================== 8) GUARDAR RESULTADOS ===================
pred_zona.to_csv("prediccion_zona.csv", index=False)
pred_por_ap.to_csv("prediccion_por_ap.csv", index=False)
print("\nArchivos guardados: prediccion_zona.csv, prediccion_por_ap.csv")

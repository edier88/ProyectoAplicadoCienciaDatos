# utils_trafico.py

import pandas as pd
import numpy as np

def ensure_datetime(df, col='fecha'):
    df[col] = pd.to_datetime(df[col])
    return df

def add_time_features(df, date_col='fecha'):
    df['dow'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    return df

def aggregate_zone_daily(df):
    """
    Suma tráfico y conexiones por zona y fecha (sin imputar aún).
    Espera columnas: fecha, zona, id_ap, conexiones, trafico_kB
    """
    agg = (df.groupby(['fecha','zona'], as_index=False)
             .agg(conexiones_totales=('conexiones','sum'),
                  trafico_total_kB=('trafico_kB','sum')))
    return agg

def compute_ap_proportions(df):
    """
    Proporción histórica por AP dentro de su zona, por día de semana (mediana robusta).
    Retorna df con columnas: zona, id_ap, dow, prop_ap
    """
    tmp = df.copy()
    tmp = ensure_datetime(tmp, 'fecha')
    tmp['dow'] = tmp['fecha'].dt.dayofweek

    # proporción del día observado
    denom = tmp.groupby(['fecha','zona'])['trafico_kB'].transform('sum')
    tmp['prop_ap'] = tmp['trafico_kB'] / denom

    # mediana por zona, ap y dow (robusto a outliers)
    prop_hist = (tmp.dropna(subset=['prop_ap'])
                   .groupby(['zona','id_ap','dow'])['prop_ap']
                   .median()
                   .reset_index())
    return prop_hist

def impute_missing_ap_by_proportion(df, prop_hist):
    """
    Imputa tráfico faltante de un AP con base en la proporción histórica por día de semana
    de su zona (mediana). Si falta un AP en la pareja (solo 2 AP), se estima a partir del otro.
    Entrada df: fecha, zona, id_ap, trafico_kB
    Retorna df con columna trafico_kB_imput y flag_imputado (0/1)
    """
    x = df.copy()
    x = ensure_datetime(x, 'fecha')
    x['dow'] = x['fecha'].dt.dayofweek

    # unir proporciones históricas
    x = x.merge(prop_hist, on=['zona','id_ap','dow'], how='left', suffixes=('','_hist'))

    # pivot para ver parejas por zona-fecha
    wide = (x.pivot_table(index=['fecha','zona','dow'], 
                          columns='id_ap', values='trafico_kB', aggfunc='first')
              .copy())

    # también pivot de proporciones
    wide_prop = (x.pivot_table(index=['fecha','zona','dow'], 
                               columns='id_ap', values='prop_ap', aggfunc='first')
                   .copy())

    # y proporciones históricas
    wide_prop_hist = (x.pivot_table(index=['fecha','zona','dow'], 
                                    columns='id_ap', values='prop_ap_hist', aggfunc='first')
                        .copy())

    # función para imputar una fila (zona-fecha)
    def _impute_row(row):
        vals = row.copy()
        # cuántos AP con dato
        notna_cols = vals[~vals.isna()].index.tolist()
        na_cols = vals[vals.isna()].index.tolist()

        # nada que imputar
        if len(na_cols) == 0:
            return vals, 0

        # si faltan los dos, no podemos imputar sin más (se imputa luego a nivel zona si quieres)
        if len(na_cols) == 2:
            # retornar NaN, flag 0 aquí (la imputación zonal vendrá después si se usa)
            return vals, 0

        # falta uno: imputar con proporción histórica del faltante (o del observado)
        # ejemplo: si conocemos AP_obs con tráfico y prop_hist_obs s_obs,
        # entonces tráfico_total ~ AP_obs / s_obs
        # y AP_missing = tráfico_total * s_missing
        ap_obs = notna_cols[0]
        ap_miss = na_cols[0]

        # proporciones históricas
        s_obs = wide_prop_hist.loc[row.name].get(ap_obs, np.nan)
        s_mis = wide_prop_hist.loc[row.name].get(ap_miss, np.nan)

        y_obs = vals[ap_obs]

        # si no hay proporción histórica, usa 0.5 y 0.5 como fallback (o media por zona)
        if pd.isna(s_obs) or pd.isna(s_mis) or s_obs <= 0 or s_mis <= 0:
            s_obs, s_mis = 0.5, 0.5

        # estimar total y el faltante
        y_total = y_obs / s_obs
        vals[ap_miss] = y_total * s_mis

        return vals, 1

    imputed_rows = []
    flags = []
    for idx, row in wide.iterrows():
        vals, flag = _impute_row(row)
        imputed_rows.append(vals)
        flags.append(flag)

    wide_imputed = pd.DataFrame(imputed_rows, index=wide.index, columns=wide.columns)
    wide_imputed['flag_imputado_ap'] = flags

    # volver a largo
    long_imp = (wide_imputed
                .reset_index()
                .melt(id_vars=['fecha','zona','dow','flag_imputado_ap'],
                      var_name='id_ap', value_name='trafico_kB_imput'))

    # unir de vuelta al original
    out = (x[['fecha','zona','id_ap','trafico_kB','dow']]
             .merge(long_imp, on=['fecha','zona','id_ap','dow'], how='left'))

    out['flag_imputado_ap'] = out['flag_imputado_ap'].fillna(0).astype(int)
    return out

def build_zone_panel_with_imputation(df):
    """
    Construye el panel por zona-fecha con suma de tráfico y conexiones,
    aplicando imputación cuando falte un solo AP.
    Retorna: zona_df con trafico_total_kB, conexiones_totales y flag_imputado_zona
    """
    # proporciones históricas
    prop_hist = compute_ap_proportions(df)

    # imputar a nivel AP cuando falta uno
    ap_imp = impute_missing_ap_by_proportion(df[['fecha','zona','id_ap','trafico_kB']].copy(), prop_hist)

    # sumar por zona-fecha (usando tráfico imputado cuando exista)
    tmp = ap_imp.copy()
    tmp['trafico_kB_final'] = tmp['trafico_kB_imput']
    # si no se imputó (ambos presentes), trafico_kB_imput = real
    # si faltaron ambos, seguirá NaN -> se manejará luego con mediana por DOW

    # flag a nivel zona si algún AP fue imputado
    flag_zona = (tmp.groupby(['fecha','zona'])['flag_imputado_ap']
                   .max()
                   .rename('flag_imputado_zona')
                   .reset_index())

    # agregar conexiones originales (sumar las disponibles; si faltan, 0 o mediana por DOW si deseas)
    conex = (df.groupby(['fecha','zona'], as_index=False)['conexiones'].sum()
               .rename(columns={'conexiones':'conexiones_totales'}))

    zona = (tmp.groupby(['fecha','zona'], as_index=False)['trafico_kB_final'].sum()
              .rename(columns={'trafico_kB_final':'trafico_total_kB'}))

    zona = zona.merge(flag_zona, on=['fecha','zona'], how='left')
    zona = zona.merge(conex, on=['fecha','zona'], how='left')

    # imputar días donde faltaron ambos AP (trafico_total_kB NaN) con mediana por DOW (robusto)
    zona = ensure_datetime(zona, 'fecha')
    zona['dow'] = zona['fecha'].dt.dayofweek
    med_dow = zona.groupby(['zona','dow'])['trafico_total_kB'].transform('median')
    missing_both = zona['trafico_total_kB'].isna()
    zona.loc[missing_both, 'trafico_total_kB'] = med_dow[missing_both]
    zona.loc[missing_both, 'flag_imputado_zona'] = 1  # marcar también como imputado

    return zona

def add_lags_sorted(df, group_col='zona', target='trafico_total_kB', lags=(1,7)):
    x = df.sort_values([group_col,'fecha']).copy()
    for L in lags:
        x[f'lag{L}'] = x.groupby(group_col)[target].shift(L)
    return x

def simple_walkforward_splits(n, n_splits=5, min_train=100):
    """
    Genera índices (train_idx, test_idx) para validación temporal simple.
    - n: número de filas totales (asumiendo df ya ordenado por fecha)
    """
    splits = []
    fold_size = n // (n_splits + 1)
    start = max(min_train, fold_size)
    for k in range(n_splits):
        train_end = start + k*fold_size
        test_end = min(train_end + fold_size, n)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) > 0 and len(train_idx) >= min_train:
            splits.append((train_idx, test_idx))
    return splits

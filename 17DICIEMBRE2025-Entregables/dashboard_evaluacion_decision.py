#!/usr/bin/env python3
"""
Dashboard de Evaluación y Decisión (nivel DATIC / EMCALI)
Este dashboard justifica inversión, no es solo técnico.

Secciones:
A. Comparación de modelos (SVR, RF, MLP, Regresión - Base vs Optimizado)
B. Ranking de zonas (menor MAPE, mejor R², zonas problemáticas)
C. Confianza para toma de decisiones (zonas confiables, zonas que requieren revisión)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Evaluación y Decisión - Zonas WiFi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Dashboard de Evaluación y Decisión")
st.subheader("Nivel DATIC / EMCALI")
st.markdown("---")

# Cargar datos
@st.cache_data
def cargar_metricas():
    """Carga todas las métricas de los modelos."""
    base_dir = Path("metricas")
    
    # Cargar archivos con Base/Optimizado
    svr = pd.read_csv(base_dir / "metricas_SVR.csv")
    rf = pd.read_csv(base_dir / "metricas_RandomForest.csv")
    mlp = pd.read_csv(base_dir / "metricas_MLP.csv")
    
    # Cargar Regresión Lineal (sin Base/Optimizado)
    rl = pd.read_csv(base_dir / "metricas_RegresionLineal.csv")
    
    # Normalizar nombres de modelos
    svr['Modelo'] = 'SVR'
    rf['Modelo'] = 'Random Forest'
    mlp['Modelo'] = 'MLP'
    rl['Modelo'] = 'Regresión Lineal'
    
    # Para RL, crear columnas compatibles (usar mismo valor para Base y Optimizado)
    rl['MAPE_Base'] = rl['MAPE']
    rl['MAPE_Optimizado'] = rl['MAPE']
    rl['MAPE(%)_Base'] = rl['MAPE(%)']
    rl['MAPE(%)_Optimizado'] = rl['MAPE(%)']
    rl['MAE_Base'] = rl['MAE']
    rl['MAE_Optimizado'] = rl['MAE']
    rl['RMSE_Base'] = rl['RMSE']
    rl['RMSE_Optimizado'] = rl['RMSE']
    rl['R2_Base'] = rl['R2']
    rl['R2_Optimizado'] = rl['R2']
    
    # Combinar todos los modelos
    df_completo = pd.concat([svr, rf, mlp, rl], ignore_index=True)
    
    # Limpiar nombre de zona (quitar .csv)
    df_completo['Zona_Limpia'] = df_completo['Zona'].str.replace('.csv', '', regex=False)
    
    return df_completo

# Cargar datos
try:
    df = cargar_metricas()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Sidebar con filtros
st.sidebar.header("🔍 Filtros")
modelos_seleccionados = st.sidebar.multiselect(
    "Seleccionar modelos:",
    options=df['Modelo'].unique(),
    default=df['Modelo'].unique()
)

# Filtrar datos
df_filtrado = df[df['Modelo'].isin(modelos_seleccionados)]

# ============================================================================
# SECCIÓN A: COMPARACIÓN DE MODELOS
# ============================================================================
st.header("A. Comparación de Modelos")
st.markdown("**¿SVR, RF, MLP o Regresión funciona mejor? ¿Base vs Optimizado?**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Comparación por Métrica (Optimizado)")
    
    # Seleccionar métrica
    metrica = st.selectbox(
        "Seleccionar métrica:",
        options=['MAPE(%)', 'MAE', 'RMSE', 'R²'],
        key='metrica_comparacion'
    )
    
    # Mapear nombre de métrica a columna
    columna_metrica = {
        'MAPE(%)': 'MAPE(%)_Optimizado',
        'MAE': 'MAE_Optimizado',
        'RMSE': 'RMSE_Optimizado',
        'R²': 'R2_Optimizado'
    }[metrica]
    
    # Calcular promedio por modelo
    df_agrupado = df_filtrado.groupby('Modelo')[columna_metrica].agg(['mean', 'std']).reset_index()
    df_agrupado.columns = ['Modelo', 'Promedio', 'Desviación']
    
    # Para R², mayor es mejor; para el resto, menor es mejor
    ordenar_asc = metrica != 'R²'
    df_agrupado = df_agrupado.sort_values('Promedio', ascending=ordenar_asc)
    
    # Gráfico de barras
    fig = px.bar(
        df_agrupado,
        x='Modelo',
        y='Promedio',
        error_y='Desviación',
        title=f'Promedio de {metrica} por Modelo (Optimizado)',
        labels={'Promedio': f'{metrica} Promedio', 'Modelo': 'Modelo'},
        color='Promedio',
        color_continuous_scale='RdYlGn' if metrica == 'R²' else 'RdYlGn_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla resumen
    st.dataframe(
        df_agrupado.style.format({
            'Promedio': '{:.3f}',
            'Desviación': '{:.3f}'
        }),
        use_container_width=True
    )

with col2:
    st.subheader("🔄 Mejora Base vs Optimizado")
    
    # Calcular mejora (diferencia porcentual)
    df_filtrado['Mejora_MAPE(%)'] = ((df_filtrado['MAPE(%)_Base'] - df_filtrado['MAPE(%)_Optimizado']) / df_filtrado['MAPE(%)_Base']) * 100
    df_filtrado['Mejora_R2'] = df_filtrado['R2_Optimizado'] - df_filtrado['R2_Base']
    
    # Agrupar por modelo
    mejora_por_modelo = df_filtrado.groupby('Modelo').agg({
        'Mejora_MAPE(%)': 'mean',
        'Mejora_R2': 'mean'
    }).reset_index()
    
    # Gráfico de mejora MAPE
    fig_mejora = go.Figure()
    fig_mejora.add_trace(go.Bar(
        x=mejora_por_modelo['Modelo'],
        y=mejora_por_modelo['Mejora_MAPE(%)'],
        name='Mejora MAPE (%)',
        marker_color='lightblue'
    ))
    fig_mejora.update_layout(
        title='Mejora Promedio MAPE (%) - Base vs Optimizado',
        xaxis_title='Modelo',
        yaxis_title='Mejora (%)',
        height=400
    )
    st.plotly_chart(fig_mejora, use_container_width=True)
    
    # Gráfico de mejora R²
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(
        x=mejora_por_modelo['Modelo'],
        y=mejora_por_modelo['Mejora_R2'],
        name='Mejora R²',
        marker_color='lightgreen'
    ))
    fig_r2.update_layout(
        title='Mejora Promedio R² - Base vs Optimizado',
        xaxis_title='Modelo',
        yaxis_title='Mejora R²',
        height=400
    )
    st.plotly_chart(fig_r2, use_container_width=True)

# Tabla comparativa detallada
st.subheader("📋 Tabla Comparativa Detallada")
st.markdown("Comparación de todos los modelos por zona (versión optimizada)")

# Crear tabla comparativa
df_comparativo = df_filtrado.pivot_table(
    index='Zona_Limpia',
    columns='Modelo',
    values=['MAPE(%)_Optimizado', 'R2_Optimizado'],
    aggfunc='first'
)

# Mostrar tabla interactiva
st.dataframe(
    df_comparativo.style.format('{:.2f}'),
    use_container_width=True,
    height=400
)

st.markdown("---")

# ============================================================================
# SECCIÓN B: RANKING DE ZONAS
# ============================================================================
st.header("B. Ranking de Zonas")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏆 Zonas con Menor MAPE")
    modelo_ranking = st.selectbox(
        "Modelo para ranking:",
        options=df_filtrado['Modelo'].unique(),
        key='modelo_ranking_mape'
    )
    
    df_ranking_mape = df_filtrado[df_filtrado['Modelo'] == modelo_ranking].copy()
    df_ranking_mape = df_ranking_mape.sort_values('MAPE(%)_Optimizado').head(10)
    
    fig_mape = px.bar(
        df_ranking_mape,
        x='MAPE(%)_Optimizado',
        y='Zona_Limpia',
        orientation='h',
        title=f'Top 10 Zonas con Menor MAPE - {modelo_ranking}',
        labels={'MAPE(%)_Optimizado': 'MAPE (%)', 'Zona_Limpia': 'Zona'},
        color='MAPE(%)_Optimizado',
        color_continuous_scale='Greens_r'
    )
    fig_mape.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_mape, use_container_width=True)
    
    st.dataframe(
        df_ranking_mape[['Zona_Limpia', 'MAPE(%)_Optimizado', 'R2_Optimizado']].style.format({
            'MAPE(%)_Optimizado': '{:.2f}%',
            'R2_Optimizado': '{:.3f}'
        }),
        use_container_width=True
    )

with col2:
    st.subheader("⭐ Zonas con Mejor R²")
    modelo_ranking_r2 = st.selectbox(
        "Modelo para ranking:",
        options=df_filtrado['Modelo'].unique(),
        key='modelo_ranking_r2'
    )
    
    df_ranking_r2 = df_filtrado[df_filtrado['Modelo'] == modelo_ranking_r2].copy()
    df_ranking_r2 = df_ranking_r2.sort_values('R2_Optimizado', ascending=False).head(10)
    
    fig_r2_ranking = px.bar(
        df_ranking_r2,
        x='R2_Optimizado',
        y='Zona_Limpia',
        orientation='h',
        title=f'Top 10 Zonas con Mejor R² - {modelo_ranking_r2}',
        labels={'R2_Optimizado': 'R²', 'Zona_Limpia': 'Zona'},
        color='R2_Optimizado',
        color_continuous_scale='Blues'
    )
    fig_r2_ranking.update_layout(height=500, yaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig_r2_ranking, use_container_width=True)
    
    st.dataframe(
        df_ranking_r2[['Zona_Limpia', 'R2_Optimizado', 'MAPE(%)_Optimizado']].style.format({
            'R2_Optimizado': '{:.3f}',
            'MAPE(%)_Optimizado': '{:.2f}%'
        }),
        use_container_width=True
    )

with col3:
    st.subheader("⚠️ Zonas Problemáticas (Alto Error)")
    modelo_problemas = st.selectbox(
        "Modelo para análisis:",
        options=df_filtrado['Modelo'].unique(),
        key='modelo_problemas'
    )
    
    # Zonas con MAPE alto o R² bajo/negativo
    df_problemas = df_filtrado[df_filtrado['Modelo'] == modelo_problemas].copy()
    df_problemas = df_problemas[
        (df_problemas['MAPE(%)_Optimizado'] > 30) | 
        (df_problemas['R2_Optimizado'] < 0.5)
    ].sort_values('MAPE(%)_Optimizado', ascending=False).head(10)
    
    if len(df_problemas) > 0:
        fig_problemas = px.scatter(
            df_problemas,
            x='MAPE(%)_Optimizado',
            y='R2_Optimizado',
            size='MAE_Optimizado',
            hover_name='Zona_Limpia',
            title=f'Zonas Problemáticas - {modelo_problemas}',
            labels={'MAPE(%)_Optimizado': 'MAPE (%)', 'R2_Optimizado': 'R²'},
            color='MAPE(%)_Optimizado',
            color_continuous_scale='Reds'
        )
        fig_problemas.update_layout(height=500)
        st.plotly_chart(fig_problemas, use_container_width=True)
        
        st.dataframe(
            df_problemas[['Zona_Limpia', 'MAPE(%)_Optimizado', 'R2_Optimizado', 'MAE_Optimizado']].style.format({
                'MAPE(%)_Optimizado': '{:.2f}%',
                'R2_Optimizado': '{:.3f}',
                'MAE_Optimizado': '{:,.0f}'
            }),
            use_container_width=True
        )
    else:
        st.info("✅ No se encontraron zonas problemáticas con los criterios seleccionados.")

st.markdown("---")

# ============================================================================
# SECCIÓN C: CONFIANZA PARA TOMA DE DECISIONES
# ============================================================================
st.header("C. Confianza para Toma de Decisiones")

# Definir criterios de confianza
umbral_mape_bueno = st.sidebar.slider("Umbral MAPE para zona confiable (%)", 0, 50, 20)
umbral_r2_bueno = st.sidebar.slider("Umbral R² para zona confiable", 0.0, 1.0, 0.7)

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Zonas Confiables para Planificación")
    st.markdown(f"**Criterios:** MAPE ≤ {umbral_mape_bueno}% y R² ≥ {umbral_r2_bueno}")
    
    modelo_confianza = st.selectbox(
        "Modelo para análisis de confianza:",
        options=df_filtrado['Modelo'].unique(),
        key='modelo_confianza'
    )
    
    df_confianza = df_filtrado[df_filtrado['Modelo'] == modelo_confianza].copy()
    df_confiables = df_confianza[
        (df_confianza['MAPE(%)_Optimizado'] <= umbral_mape_bueno) &
        (df_confianza['R2_Optimizado'] >= umbral_r2_bueno)
    ].sort_values('MAPE(%)_Optimizado')
    
    st.metric("Total de zonas confiables", len(df_confiables), f"{len(df_confiables)/len(df_confianza)*100:.1f}%")
    
    if len(df_confiables) > 0:
        fig_confiables = px.scatter(
            df_confiables,
            x='MAPE(%)_Optimizado',
            y='R2_Optimizado',
            hover_name='Zona_Limpia',
            title=f'Zonas Confiables - {modelo_confianza}',
            labels={'MAPE(%)_Optimizado': 'MAPE (%)', 'R2_Optimizado': 'R²'},
            color='MAPE(%)_Optimizado',
            color_continuous_scale='Greens_r'
        )
        fig_confiables.add_hline(y=umbral_r2_bueno, line_dash="dash", line_color="red", 
                                annotation_text=f"R² mínimo: {umbral_r2_bueno}")
        fig_confiables.add_vline(x=umbral_mape_bueno, line_dash="dash", line_color="red",
                                annotation_text=f"MAPE máximo: {umbral_mape_bueno}%")
        fig_confiables.update_layout(height=500)
        st.plotly_chart(fig_confiables, use_container_width=True)
        
        st.dataframe(
            df_confiables[['Zona_Limpia', 'MAPE(%)_Optimizado', 'R2_Optimizado', 'MAE_Optimizado']].style.format({
                'MAPE(%)_Optimizado': '{:.2f}%',
                'R2_Optimizado': '{:.3f}',
                'MAE_Optimizado': '{:,.0f}'
            }),
            use_container_width=True
        )
    else:
        st.warning("No se encontraron zonas que cumplan los criterios de confianza seleccionados.")

with col2:
    st.subheader("🔍 Zonas que Requieren Más Datos / Revisión")
    st.markdown(f"**Criterios:** MAPE > {umbral_mape_bueno}% o R² < {umbral_r2_bueno}")
    
    df_revision = df_confianza[
        (df_confianza['MAPE(%)_Optimizado'] > umbral_mape_bueno) |
        (df_confianza['R2_Optimizado'] < umbral_r2_bueno)
    ].sort_values('MAPE(%)_Optimizado', ascending=False)
    
    st.metric("Total de zonas que requieren revisión", len(df_revision), f"{len(df_revision)/len(df_confianza)*100:.1f}%")
    
    if len(df_revision) > 0:
        fig_revision = px.scatter(
            df_revision,
            x='MAPE(%)_Optimizado',
            y='R2_Optimizado',
            hover_name='Zona_Limpia',
            title=f'Zonas que Requieren Revisión - {modelo_confianza}',
            labels={'MAPE(%)_Optimizado': 'MAPE (%)', 'R2_Optimizado': 'R²'},
            color='MAPE(%)_Optimizado',
            color_continuous_scale='Reds'
        )
        fig_revision.add_hline(y=umbral_r2_bueno, line_dash="dash", line_color="blue", 
                              annotation_text=f"R² mínimo: {umbral_r2_bueno}")
        fig_revision.add_vline(x=umbral_mape_bueno, line_dash="dash", line_color="blue",
                              annotation_text=f"MAPE máximo: {umbral_mape_bueno}%")
        fig_revision.update_layout(height=500)
        st.plotly_chart(fig_revision, use_container_width=True)
        
        st.dataframe(
            df_revision[['Zona_Limpia', 'MAPE(%)_Optimizado', 'R2_Optimizado', 'MAE_Optimizado']].style.format({
                'MAPE(%)_Optimizado': '{:.2f}%',
                'R2_Optimizado': '{:.3f}',
                'MAE_Optimizado': '{:,.0f}'
            }),
            use_container_width=True
        )
    else:
        st.success("✅ Todas las zonas cumplen los criterios de confianza.")

# Resumen ejecutivo
st.markdown("---")
st.header("📊 Resumen Ejecutivo")

col1, col2, col3, col4 = st.columns(4)

modelo_resumen = st.selectbox(
    "Modelo para resumen:",
    options=df_filtrado['Modelo'].unique(),
    key='modelo_resumen'
)

df_resumen = df_filtrado[df_filtrado['Modelo'] == modelo_resumen].copy()

with col1:
    st.metric(
        "MAPE Promedio",
        f"{df_resumen['MAPE(%)_Optimizado'].mean():.2f}%"
    )

with col2:
    st.metric(
        "R² Promedio",
        f"{df_resumen['R2_Optimizado'].mean():.3f}"
    )

with col3:
    zonas_confiables_resumen = len(df_resumen[
        (df_resumen['MAPE(%)_Optimizado'] <= umbral_mape_bueno) &
        (df_resumen['R2_Optimizado'] >= umbral_r2_bueno)
    ])
    st.metric(
        "Zonas Confiables",
        zonas_confiables_resumen,
        f"{zonas_confiables_resumen/len(df_resumen)*100:.1f}%"
    )

with col4:
    st.metric(
        "Total Zonas",
        len(df_resumen)
    )

# Footer
st.markdown("---")
st.markdown("**Este dashboard justifica inversión, no es solo técnico.**")
st.caption("Dashboard de Evaluación y Decisión - Nivel DATIC / EMCALI")


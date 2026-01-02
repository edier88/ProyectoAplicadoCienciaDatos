#!/usr/bin/env python3
"""
Dashboard Proyecto Predicción Zonas WiFi de Cali

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
import glob
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Proyecto Predicción Zonas WiFi de Cali",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Dashboard Proyecto Predicción Zonas WiFi de Cali")
st.markdown("---")

# Función auxiliar para leer CSV con manejo de codificación
def leer_csv_con_codificacion(archivo):
    """Intenta leer CSV con diferentes codificaciones."""
    codificaciones = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
    for encoding in codificaciones:
        try:
            return pd.read_csv(archivo, encoding=encoding)
            #return pd.read_csv(archivo)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as e:
            continue
        except Exception as e:
            # Si es otro tipo de error, intentar con la siguiente codificación
            continue
    # Si ninguna funciona, intentar con errors='ignore' o 'replace'
    try:
        return pd.read_csv(archivo, encoding='latin-1', errors='replace')
        #return pd.read_csv(archivo)
    except:
        return pd.read_csv(archivo, encoding='utf-8', errors='replace')
        #return pd.read_csv(archivo)

# Cargar datos
@st.cache_data
def cargar_metricas():
    """Carga todas las métricas de los modelos."""
    base_dir = Path("metricas_por_modelo")
    
    # Cargar archivos con Base/Optimizado
    svr = leer_csv_con_codificacion(base_dir / "metricas_SVR.csv")
    rf = leer_csv_con_codificacion(base_dir / "metricas_RandomForest.csv")
    mlp = leer_csv_con_codificacion(base_dir / "metricas_MLP.csv")
    
    # Cargar Regresión Lineal (sin Base/Optimizado)
    rl = leer_csv_con_codificacion(base_dir / "metricas_RegresionLineal.csv")
    
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

# Función para cargar datos de mejores modelos globales
@st.cache_data
def cargar_mejores_modelos_global():
    """Carga los datos de mejores modelos estadísticos globales."""
    base_dir = Path("metricas_por_modelo")
    df = leer_csv_con_codificacion(base_dir / "mejores_modelos_estadistica_global.csv")
    return df

# Función para cargar datos de mejores modelos por zona
@st.cache_data
def cargar_mejores_modelos_por_zona():
    """Carga los datos de mejores modelos por zona."""
    base_dir = Path("metricas_por_modelo")
    df = leer_csv_con_codificacion(base_dir / "mejores_modelos_por_zona.csv")
    
    # Convertir MAPE de formato con coma a float
    # El CSV tiene formato "11,7545" que necesita convertirse a 11.7545
    df['MAPE_NUM'] = df['MAPE(%)'].str.replace(',', '.').astype(float)
    
    return df

# Cargar datos
try:
    df = cargar_metricas()
    df_mejores_global = cargar_mejores_modelos_global()
    df_mejores_zona = cargar_mejores_modelos_por_zona()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Sidebar con filtros y navegación
st.sidebar.header("🔍 Filtros")
modelos_seleccionados = st.sidebar.multiselect(
    "Seleccionar modelos:",
    options=df['Modelo'].unique(),
    default=df['Modelo'].unique()
)

# Separador
st.sidebar.markdown("---")

# Sección de gráficas en el sidebar
st.sidebar.header("📊 Visualización de Gráficas")

# Función para extraer el nombre base de una imagen (sin sufijos)
def extraer_nombre_base(nombre_archivo):
    """Extrae el nombre base de una imagen eliminando sufijos como _prediccion_7dias o _serie"""
    nombre = Path(nombre_archivo).stem
    # Eliminar sufijos comunes
    nombre = nombre.replace('_prediccion_7dias', '')
    nombre = nombre.replace('_serie', '')
    return nombre

# Función para cargar y emparejar imágenes de ambas carpetas
@st.cache_data
def cargar_imagenes_emparejadas():
    """Carga y empareja imágenes de Futuras y TrainTest por nombre de zona."""
    base_dir = Path("metricas_por_modelo")
    
    # Rutas de las carpetas
    carpeta_futuras = base_dir / "Graficas_futuras_por_mejor_modelo"
    carpeta_traintest = base_dir / "Graficas_TrainTest_por_mejor_modelo"
    
    # Cargar imágenes de ambas carpetas
    imagenes_futuras = {}
    imagenes_traintest = {}
    
    # Buscar en Futuras
    if carpeta_futuras.exists():
        for img_path in carpeta_futuras.glob('**/*.png'):
            nombre_base = extraer_nombre_base(img_path.name)
            imagenes_futuras[nombre_base] = img_path
    
    # Buscar en TrainTest
    if carpeta_traintest.exists():
        for img_path in carpeta_traintest.glob('**/*.png'):
            nombre_base = extraer_nombre_base(img_path.name)
            imagenes_traintest[nombre_base] = img_path
    
    # Emparejar imágenes que existan en ambas carpetas
    pares_imagenes = []
    nombres_comunes = set(imagenes_futuras.keys()) & set(imagenes_traintest.keys())
    
    for nombre_base in sorted(nombres_comunes):
        par = {
            'nombre_base': nombre_base,
            'imagen_futuras': imagenes_futuras[nombre_base],
            'imagen_traintest': imagenes_traintest[nombre_base]
        }
        pares_imagenes.append(par)
    
    return pares_imagenes

# Opciones de visualización de gráficas
opciones_graficas = {
    "General": False,
    "Predicciones Por Zona": True
}

carpeta_seleccionada = st.sidebar.selectbox(
    "Seleccionar visualización de gráficas:",
    options=list(opciones_graficas.keys())
)

# Filtrar datos
df_filtrado = df[df['Modelo'].isin(modelos_seleccionados)]

# Si se selecciona visualización de gráficas, mostrar el visor
if opciones_graficas[carpeta_seleccionada]:
    pares_imagenes = cargar_imagenes_emparejadas()
    
    # Inicializar o resetear índice si cambió la opción
    if 'mostrando_graficas' not in st.session_state:
        st.session_state.mostrando_graficas = True
        st.session_state.indice_par = 0
    elif not st.session_state.get('mostrando_graficas', False):
        st.session_state.mostrando_graficas = True
        st.session_state.indice_par = 0
    
    if pares_imagenes:
        # Asegurar que el índice esté dentro del rango válido
        if st.session_state.indice_par >= len(pares_imagenes):
            st.session_state.indice_par = 0
        
        # Título de la sección de gráficas
        st.header("📊 Visualizador de Predicciones por Zona")

        st.metric("Total Zonas", len(pares_imagenes))
        
        # Selector de zona tipo <select> HTML - DEBE IR ANTES DE MOSTRAR IMÁGENES
        st.subheader("📋 Seleccionar Zona")
        
        # Crear lista de opciones con nombres limpios
        opciones_zonas = []
        for par in pares_imagenes:
            nombre_par_limpio = par['nombre_base'].replace('.csv', '')
            opciones_zonas.append(nombre_par_limpio)
        
        # Selectbox para elegir zona - usar índice directamente
        indice_seleccionado = st.selectbox(
            "Selecciona una zona para ver su predicción en Serie Temporal:",
            options=range(len(opciones_zonas)),
            format_func=lambda x: opciones_zonas[x],
            index=st.session_state.indice_par,
            key="selectbox_zona"
        )
        
        # Actualizar índice si se seleccionó una zona diferente
        if indice_seleccionado != st.session_state.indice_par:
            st.session_state.indice_par = indice_seleccionado
            st.rerun()
        
        # Ahora obtener el par actual con el índice actualizado
        par_actual = pares_imagenes[st.session_state.indice_par]
        
        # Limpiar nombre de zona (eliminar extensión .csv si existe)
        nombre_zona_limpio = par_actual['nombre_base'].replace('.csv', '')
        
        # Botones de navegación anterior/siguiente
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        
        # Mostrar ambas imágenes (una arriba de la otra, como en el ejemplo)

        zonas_y_modelos = {
            "001_ZW Parque Ingenio": "Random Forest",
            "002_ZW Canchas Panamericanas": "SVR",
            "003_ZW Parque del Perro": "SVR",
            "004_ZW Parque San Nicolas": "SVR",
            "005_ZW Parque Barrio Obrero": "MLP",
            "007_ZW Parque Pizamos": "Regresion Lineal",
            "008_ZW Parque Alfonso Lopez": "SVR",
            "010_ZW Parque Antonio Nariño": "SVR",
            "011_ZW Parque Santa Rosa Poblado": "Random Forest",
            "012_ZW Polideportivo Los Naranjos": "Random Forest",
            "013_ZW Parque Llano Verde": "SVR",
            "015_ZW Centro Cultural Comuna 13": "Random Forest",
            "016_ZW Conjunto Habitacional Ramali": "Regresion Lineal",
            "019_ZW Parque Skate Board": "Random Forest",
            "021_ZW Parque Alfonso Barberena": "SVR",
            "022_ZW Polideportivo Los Farallones": "Random Forest",
            "023_ZW Parque Los Guerreros": "Random Forest",
            "024_ZW Museo La Tertulia": "Random Forest",
            "025_ZW Parque San Marino": "Random Forest",
            "026_ZW Parque La Flora": "SVR",
            "027_ZW Parque Alfonso Bonilla Aragón": "SVR",
            "028_ZW Parque Yo Amo a Siloé": "Regresion Lineal",
            "029_ZW Centro Cultural Vista Hermosa": "Random Forest",
            "030_ZW Parque Mutis": "SVR",
            "031_ZW Cancha Los Azules": "MLP",
            "032_ZW Parque La Orqueta": "Random Forest",
            "034_ZW Parque Villa Colombia": "Random Forest",
            "035_ZW Parque Colseguros": "SVR",
            "036_ZW Parque India Elena": "SVR",
            "037_ZW Parque Tequendama": "Random Forest",
            "038_ZW Parque Sector Amarillo Skate Park": "Random Forest",
            "039_ZW Biblioteca Daniel Guillard": "MLP",
            "040_ZW Polideportivo Torres Comfandi": "Random Forest",
            "041_ZW Parque Junin": "Random Forest",
            "042_ZW Parque Mariano Ramos": "SVR",
            "043_ZW Polideportivo Ricardo Balcazar": "MLP",
            "044_ZW Polideportivo San Benito": "Random Forest",
            "045_ZW Parque Santa Anita": "Random Forest",
            "046_ZW Sebastian Belalcazar": "Random Forest",
            "047_ZW Parque del Mico": "Regresion Lineal",
            "048_ZW Parque Cien Palos": "Regresion Lineal",
            "049_ZW Cerro 3 Cruces": "SVR",
            "050_ZW Parque Calima": "Regresion Lineal",
            "051_ZW Parque La Merced": "Random Forest",
            "052_ZW Puente de Colores": "Random Forest",
            "053_ZW Polideportivo Laureano Gomez": "Random Forest",
            "054_ZW El Diamante": "Random Forest",
            "055_ZW Polideportivo Petecuy": "Random Forest",
            "056_ZW Comuna 16": "Random Forest",
            "057_La Castilla": "Random Forest",
            "058_La Elvira": "Random Forest",
            "059_El Saladito": "Random Forest",
        }
        
        try:
            # Primera imagen: Gráfica Futuras (Predicción 7 días)
            img_futuras = Image.open(par_actual['imagen_futuras'])
            st.subheader("🔮 Predicción Próxima Semana")
            #fileName = par_actual['imagen_futuras'].name[4:]
            # Corta la parte final del nombre del archivo ".csv_prediccion_7dias.png"
            final_part_name = par_actual['imagen_futuras'].name.find(".csv_prediccion_7dias.png")
            img_main_name = par_actual['imagen_futuras'].name[:final_part_name]
            st.write(f"Mejor modelo encontrado para esta zona: ___{zonas_y_modelos[img_main_name]}___")
            # Se crea la imagen y su descipcion debajo
            st.image(img_futuras, caption=f"{img_main_name} - Prediccion Próxima Semana", use_container_width=True)
            
            # Segunda imagen: Gráfica Train/Test
            img_traintest = Image.open(par_actual['imagen_traintest'])
            st.subheader("📊 Comparación Predicción Test")
            # Corta la parte final del nombre del archivo ".csv_prediccion_7dias.png"
            final_part_name = par_actual['imagen_traintest'].name.find(".csv_serie.png")
            img_main_name = par_actual['imagen_traintest'].name[:final_part_name]
            # Se crea la imagen y su descipcion debajo
            st.image(img_traintest, caption=f"{img_main_name} - Prediccion vs Train/Test", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al cargar las imágenes: {e}")
            if par_actual['imagen_futuras'].exists():
                st.info(f"Imagen Futuras: {par_actual['imagen_futuras']}")
            else:
                st.warning(f"Imagen Futuras no encontrada: {par_actual['imagen_futuras']}")
            if par_actual['imagen_traintest'].exists():
                st.info(f"Imagen TrainTest: {par_actual['imagen_traintest']}")
            else:
                st.warning(f"Imagen TrainTest no encontrada: {par_actual['imagen_traintest']}")
        
        # No mostrar el resto del dashboard cuando se están viendo gráficas
        st.stop()
    else:
        st.warning("⚠️ No se encontraron pares de imágenes emparejadas.")
        st.info("💡 Verifica que ambas carpetas existan y contengan imágenes PNG con nombres compatibles.")
else:
    # Si no se están mostrando gráficas, resetear el flag
    st.session_state.mostrando_graficas = False

# ============================================================================
# SECCIÓN A: COMPARACIÓN DE MODELOS
# ============================================================================
st.header("Comparación de Modelos Base con Optimizados")

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
st.header("Ranking de Zonas")

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

# Definir criterios de confianza
umbral_mape_bueno = st.sidebar.slider("Umbral MAPE para zona confiable (%)", 0, 50, 20)
umbral_r2_bueno = st.sidebar.slider("Umbral R² para zona confiable", 0.0, 1.0, 0.7)


# ============================================================================
# SECCIÓN D: ANÁLISIS DE MEJORES MODELOS
# ============================================================================
st.header("Análisis de Mejores Modelos")

st.subheader("🥧 Distribución Global de Mejores Modelos")
    
# Gráfico de torta
fig_torta = px.pie(
    df_mejores_global,
    values='Total Zonas Con Mejor Comportamiento',
    names='Tecnica',
    title='Proporción de Mejores Modelos por Zona',
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_torta.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hovertemplate='<b>%{label}</b><br>Zonas: %{value}<br>Porcentaje: %{percent}<extra></extra>'
)
fig_torta.update_layout(height=500)
st.plotly_chart(fig_torta, use_container_width=True)

# Descripción del gráfico
with st.expander("ℹ️ Explicación del gráfico"):
    st.markdown("""
    Este gráfico de torta muestra la proporción de veces que cada modelo estadístico fue identificado como el de mejor desempeño en el análisis global realizado. Cada segmento representa el porcentaje de zonas o evaluaciones en las que un modelo resultó superior frente a los demás, según las métricas de desempeño establecidas.
    
    La visualización permite identificar rápidamente qué modelos dominan el análisis, cuáles tienen un desempeño competitivo y cuáles presentan una participación menor, facilitando la comparación global y la toma de decisiones sobre qué enfoques estadísticos son más robustos para el problema estudiado.
    """)

# Tabla de datos
st.dataframe(
    df_mejores_global.style.format({
        'Total Zonas Con Mejor Comportamiento': '{:.0f}'
    }),
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Top 10 Zonas con Mayor MAPE")
    
    # Obtener top 10 zonas con menor MAPE
    df_peores10 = df_mejores_zona.nlargest(10, 'MAPE_NUM').copy()
    df_peores10 = df_peores10.sort_values('MAPE_NUM', ascending=False)
    
    # Gráfico de barras
    fig_barras1 = px.bar(
        df_peores10,
        x='MAPE_NUM',
        y='ZONA',
        orientation='h',
        color='MEJOR_MODELO_ENCONTRADO',
        title='Top 10 Zonas WiFi con Mayor MAPE',
        labels={
            'MAPE_NUM': 'MAPE (%)',
            'ZONA': 'Zona',
            'MEJOR_MODELO_ENCONTRADO': 'Modelo Utilizado'
        },
        color_discrete_map={
            'Random Forest': '#1f77b4',
            'SVR': '#ff7f0e',
            'Regresion Lineal': '#2ca02c',
            'Perceptron': '#d62728'
        }
    )
    fig_barras1.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='MAPE (%)',
        yaxis_title='Zona'
    )
    st.plotly_chart(fig_barras1, use_container_width=True)
    
    # Descripción del gráfico
    with st.expander("ℹ️ Explicación del gráfico"):
        st.markdown("""
        Este gráfico de barras presenta las 10 zonas WiFi con mayor MAPE, es decir, aquellas donde la predicción del tráfico fue menos precisa. Cada barra representa una zona específica, y su tamaño corresponde directamente al valor del MAPE obtenido. Adicionalmente, se identifica el modelo estadístico utilizado en cada zona, permitiendo evaluar qué enfoques generan mejores resultados. La visualización facilita la comparación de precisión entre zonas y apoya la selección de modelos más confiables para la toma de decisiones.
        """)
    
    # Tabla de datos (top 10) - crear columna formateada para mostrar
    #df_top10_display = df_top10[['ZONA', 'MAPE_NUM', 'MEJOR_MODELO_ENCONTRADO']].copy()
    #df_top10_display['MAPE (%)'] = df_top10_display['MAPE_NUM'].apply(lambda x: f"{x:.2f}%")
    #df_top10_display = df_top10_display[['ZONA', 'MAPE (%)', 'MEJOR_MODELO_ENCONTRADO']]
    #df_top10_display.columns = ['Zona', 'MAPE (%)', 'Modelo Utilizado']

    # Tabla de datos - crear columna formateada para mostrar
    df_mejores_zona_display = df_mejores_zona[['ZONA', 'MAPE_NUM', 'MEJOR_MODELO_ENCONTRADO']].copy()
    df_peores_zona_display = df_mejores_zona_display.sort_values(by='MAPE_NUM', ascending=False)
    df_peores_zona_display['MAPE (%)'] = df_peores_zona_display['MAPE_NUM'].apply(lambda x: f"{x:.2f}%")
    df_peores_zona_display = df_peores_zona_display[['ZONA', 'MAPE (%)', 'MEJOR_MODELO_ENCONTRADO']]
    df_peores_zona_display.columns = ['Zona', 'MAPE (%)', 'Modelo Utilizado']
    

    
    st.dataframe(
        df_peores_zona_display,
        use_container_width=True
    )
with col2:
    st.subheader("📊 Top 10 Zonas con Menor MAPE")
    
    # Obtener top 10 zonas con menor MAPE
    df_top10 = df_mejores_zona.nsmallest(10, 'MAPE_NUM').copy()
    df_top10 = df_top10.sort_values('MAPE_NUM', ascending=False)
    
    # Gráfico de barras
    fig_barras2 = px.bar(
        df_top10,
        x='MAPE_NUM',
        y='ZONA',
        orientation='h',
        color='MEJOR_MODELO_ENCONTRADO',
        title='Top 10 Zonas WiFi con Menor MAPE',
        labels={
            'MAPE_NUM': 'MAPE (%)',
            'ZONA': 'Zona',
            'MEJOR_MODELO_ENCONTRADO': 'Modelo Utilizado'
        },
        color_discrete_map={
            'Random Forest': '#1f77b4',
            'SVR': '#ff7f0e',
            'Regresion Lineal': '#2ca02c',
            'Perceptron': '#d62728'
        }
    )
    fig_barras2.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='MAPE (%)',
        yaxis_title='Zona '
    )
    st.plotly_chart(fig_barras2, use_container_width=True)
    
    # Descripción del gráfico
    with st.expander("ℹ️ Explicación del gráfico"):
        st.markdown("""
        Este gráfico de barras presenta las 10 zonas WiFi con menor MAPE, es decir, aquellas donde la predicción del tráfico fue más precisa. Cada barra representa una zona específica, y su tamaño corresponde directamente al valor del MAPE obtenido. Adicionalmente, se identifica el modelo estadístico utilizado en cada zona, permitiendo evaluar qué enfoques generan mejores resultados. La visualización facilita la comparación de precisión entre zonas y apoya la selección de modelos más confiables para la toma de decisiones.
        """)
    
    # Tabla de datos (top 10) - crear columna formateada para mostrar
    #df_top10_display = df_top10[['ZONA', 'MAPE_NUM', 'MEJOR_MODELO_ENCONTRADO']].copy()
    #df_top10_display['MAPE (%)'] = df_top10_display['MAPE_NUM'].apply(lambda x: f"{x:.2f}%")
    #df_top10_display = df_top10_display[['ZONA', 'MAPE (%)', 'MEJOR_MODELO_ENCONTRADO']]
    #df_top10_display.columns = ['Zona', 'MAPE (%)', 'Modelo Utilizado']

    # Tabla de datos - crear columna formateada para mostrar
    df_mejores_zona_display = df_mejores_zona[['ZONA', 'MAPE_NUM', 'MEJOR_MODELO_ENCONTRADO']].copy()
    df_mejores_zona_display['MAPE (%)'] = df_mejores_zona_display['MAPE_NUM'].apply(lambda x: f"{x:.2f}%")
    df_mejores_zona_display = df_mejores_zona_display[['ZONA', 'MAPE (%)', 'MEJOR_MODELO_ENCONTRADO']]
    df_mejores_zona_display.columns = ['Zona', 'MAPE (%)', 'Modelo Utilizado']
    
    st.dataframe(
        df_mejores_zona_display,
        use_container_width=True
    )

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

st.write("Las zonas con mejores predicciones son las que tienen un MAPE bajo. Estas zonas son, desde el punto de vista del negocio, las más rentables para promocionar productos y servicios a más personas. Desde el punto de vista de lo público son zonas adecuadas para proyectar eventos con anticipación y prever si necesita más recursos de ancho de banda o Hardware para su operación, o si por el contrario necesita menos, esto podría indicar que la inversión en dicha zona no debe ser tanta y se debe enfocar en otras zonas.")
st.write("Como se muestra en el dashboard solo 6 zonas tienen un MAPE mayor al 40%. Lo que indica que solo un 11.5% del total de las zonas no son buenas para la predicción")
st.write("El mal desempeño de la predicción en el 11.5% de las zonas se debe a que el tráfico de ellas es más errático, tiene picos bruscos de poco tráfico hacia los Gigabytes en poco tiempo, teniendo así un consumo de datos poco sostenido en el tiempo. Esto se puede detallar de manera gráfica en la opción 'Predicciones por Zona' del presente Dashboard")

# Footer
st.markdown("---")
st.caption("Dashboard Proyecto Predicción Zonas WiFi de Cali")



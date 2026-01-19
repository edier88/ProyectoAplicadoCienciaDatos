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
import joblib
import matplotlib.pyplot as plt

MODELOS_DIR = Path("modelos_guardados")
ZONAS_DIR = Path("csv-zonas-wifi-separados-man-renumerados")

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

def crear_lags_exogenos(df_futuro, df_pasado, lags, steps):
    """Crea lags para las variables de entrada del modelo (variables exogenas ventaneadas 10 días atrás)"""

    exog_variables_lags = list(df_futuro.columns)

    # Se crean los titulos de las columnas del dataframe ventaneado (con lags)
    for var in df_futuro:
        for lag in range(1, lags + 1):
            exog_variables_lags.append(f'{var}_lag_{lag}')

    #step = steps -1

    print("exog_variables_lags----------------")
    print(exog_variables_lags)
    exog_variables_lags_df = pd.DataFrame(columns=exog_variables_lags)

    df_pasado = df_pasado.drop('USAGE_KB', axis=1) # Elimino la columna USAGE_KB del dataset del pasado (ventana de 10 días)
    df_pasado_y_futuro = pd.concat([df_pasado, df_futuro], axis=0) # Junto los dataframe del pasado y del futuro para hacer uno solo
    df_pasado_y_futuro = df_pasado_y_futuro.reset_index(drop=True)

    #print("df_pasado_y_futuro:---------------")
    #print(df_pasado_y_futuro)
    #print(df_pasado_y_futuro.loc[13, "PORCENTAJE_USO"])

    #print("exog_variables_lags:---------------")
    #print(exog_variables_lags)
    #exog_variables_lags.loc[0, "DIA_SEMANA_lag_9"] = 9
    #print(exog_variables_lags["DIA_SEMANA_lag_9"])

    #df_con_lags = df.copy()
    exog_variables_list = list(df_futuro.columns)

    #error
    print("df_pasado_y_futuro:")
    print(df_pasado_y_futuro.dtypes)

    
    for step in range(0, steps):
        #print (print(f"STEP: {step}"))

        for var in exog_variables_list:
            #print(f"  {var}: lags 1 a {lags}")
            for lag in range(1, lags + 1):
                position = 11-lag+step
                col_name = f'{var}_lag_{lag}'
                #print(f"STEP: {step}")
                #print(f"COL_NAME: {col_name}")
                #print(f"VAR: {var}")
                #print(f"POSITION: {position}")
                exog_variables_lags_df.loc[step, col_name] = df_pasado_y_futuro.loc[position, var].copy()
            
    print("DF_FUTURO:")
    print(df_futuro)
    #df_futuro = df_futuro.reset_index(drop=True)
    exog_variables_lags_df.index = df_futuro.index.copy()
    exog_variables_lags_df["DIA_SEMANA"] = df_futuro["DIA_SEMANA"].copy()
    exog_variables_lags_df["LABORAL"] = df_futuro["LABORAL"].copy()
    exog_variables_lags_df["FIN_DE_SEMANA"] = df_futuro["FIN_DE_SEMANA"].copy()
    exog_variables_lags_df["FESTIVO"] = df_futuro["FESTIVO"].copy()
    exog_variables_lags_df["PORCENTAJE_USO"] = df_futuro["PORCENTAJE_USO"].copy()
    exog_variables_lags_df["NUMERO_CONEXIONES"] = df_futuro["NUMERO_CONEXIONES"].copy()
    
    # Eliminar filas con NaN
    #df_con_lags = df_con_lags.iloc[lags:].copy()

    exog_variables_lags_df = exog_variables_lags_df.apply(pd.to_numeric, errors='ignore')
    
    
    return exog_variables_lags_df, df_pasado_y_futuro, exog_variables_lags

# Función auxiliar para leer CSV con manejo de codificación
def leer_csv_con_codificacion(archivo):
    """Intenta leer CSV con diferentes codificaciones."""
    codificaciones = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
    for encoding in codificaciones:
        try:
            return pd.read_csv(archivo, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as e:
            continue
        except Exception as e:
            # Si es otro tipo de error, intentar con la siguiente codificación
            continue
    # Si ninguna funciona, intentar con latin-1 que es más tolerante
    # latin-1 puede leer cualquier byte, así que siempre funciona
    try:
        return pd.read_csv(archivo, encoding='latin-1')
    except Exception as e:
        # Último recurso: leer como latin-1 y luego limpiar caracteres problemáticos
        try:
            with open(archivo, 'rb') as f:
                content = f.read()
            # Decodificar con latin-1 y re-encodificar a utf-8 reemplazando caracteres problemáticos
            content_clean = content.decode('latin-1', errors='replace').encode('utf-8', errors='replace')
            import io
            return pd.read_csv(io.StringIO(content_clean.decode('utf-8')))
        except:
            # Si todo falla, usar latin-1 directamente
            return pd.read_csv(archivo, encoding='latin-1')

# Cargar datos
@st.cache_data
def cargar_metricas():
    """Carga todas las métricas de los modelos."""
    base_dir = Path("metricas_por_modelo")
    print(base_dir)
    
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

# Las secciones de filtros y navegación están ahora en la sección de navegación principal
# Inicializar variables por defecto
if 'modelos_seleccionados' not in locals():
    modelos_seleccionados = df['Modelo'].unique()

# ============================================================================
# NAVEGACIÓN PRINCIPAL
# ============================================================================
st.sidebar.header("🧭 Navegación")

pagina_seleccionada = st.sidebar.radio(
    "Seleccionar sección:",
    options=["🔮 Predicción Interactiva", "📊 Análisis y Métricas", "📈 Visualización de Gráficas"],
    key="pagina_principal"
)

st.sidebar.markdown("---")

# ============================================================================
# SECCIÓN: PREDICCIÓN INTERACTIVA EN SIDEBAR
# ============================================================================
if pagina_seleccionada == "🔮 Predicción Interactiva":
    st.sidebar.header("🔮 Parámetros de Predicción")
    
    # Obtener lista de zonas disponibles
    zonas_disponibles = sorted(df['Zona_Limpia'].unique().tolist())
    
    zona_seleccionada_pred = st.sidebar.selectbox(
        "Zona WiFi:",
        options=zonas_disponibles,
        key="zona_prediccion"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Parámetros")

    st.sidebar.subheader("📆 Día 1:")
    
    porcentaje_uso_1 = st.sidebar.number_input(
        "Porcentaje de Uso (%):",
        min_value=0,
        value=0,
        step=1,
        key="porcentaje_uso_1"
    )

    numero_conexiones_1 = st.sidebar.number_input(
        "Número de Conexiones:",
        min_value=0,
        value=100,
        step=1,
        key="num_conexiones_1"
    )

    st.sidebar.subheader("📆 Día 2:")

    porcentaje_uso_2 = st.sidebar.number_input(
        "Porcentaje de Uso (%):",
        min_value=0,
        value=0,
        step=1,
        key="porcentaje_uso_2"
    )
    
    numero_conexiones_2 = st.sidebar.number_input(
        "Número de Conexiones:",
        min_value=0,
        value=100,
        step=1,
        key="num_conexiones_2"
    )

    st.sidebar.subheader("📆 Día 3:")

    porcentaje_uso_3 = st.sidebar.number_input(
        "Porcentaje de Uso (%):",
        min_value=0,
        value=0,
        step=1,
        key="porcentaje_uso_3"
    )
    
    numero_conexiones_3 = st.sidebar.number_input(
        "Número de Conexiones:",
        min_value=0,
        value=100,
        step=1,
        key="num_conexiones_3"
    )

    st.sidebar.subheader("📆 Día 4:")

    porcentaje_uso_4 = st.sidebar.number_input(
        "Porcentaje de Uso (%):",
        min_value=0,
        value=0,
        step=1,
        key="porcentaje_uso_4"
    )
    
    numero_conexiones_4 = st.sidebar.number_input(
        "Número de Conexiones:",
        min_value=0,
        value=100,
        step=1,
        key="num_conexiones_4"
    )

    st.sidebar.subheader("📆 Día 5:")

    porcentaje_uso_5 = st.sidebar.number_input(
        "Porcentaje de Uso (%):",
        min_value=0,
        value=0,
        step=1,
        key="porcentaje_uso_5"
    )
    
    numero_conexiones_5 = st.sidebar.number_input(
        "Número de Conexiones:",
        min_value=0,
        value=100,
        step=1,
        key="num_conexiones_5"
    )

    st.sidebar.subheader("📆 Día 6:")

    porcentaje_uso_6 = st.sidebar.number_input(
        "Porcentaje de Uso (%):",
        min_value=0,
        value=0,
        step=1,
        key="porcentaje_uso_6"
    )
    
    numero_conexiones_6 = st.sidebar.number_input(
        "Número de Conexiones:",
        min_value=0,
        value=100,
        step=1,
        key="num_conexiones_6"
    )

    st.sidebar.subheader("📆 Día 7:")

    porcentaje_uso_7 = st.sidebar.number_input(
        "Porcentaje de Uso (%):",
        min_value=0,
        value=0,
        step=1,
        key="porcentaje_uso_7"
    )
    
    numero_conexiones_7 = st.sidebar.number_input(
        "Número de Conexiones:",
        min_value=0,
        value=100,
        step=1,
        key="num_conexiones_7"
    )
    
    # Botón para realizar predicción
    boton_prediccion = st.sidebar.button("🚀 Generar Predicción", type="primary", use_container_width=True)
else:
    boton_prediccion = False

# Filtros para otras secciones
if pagina_seleccionada == "📊 Análisis y Métricas":
    st.sidebar.header("🔍 Filtros")
    modelos_seleccionados = st.sidebar.multiselect(
        "Seleccionar modelos:",
        options=df['Modelo'].unique(),
        default=df['Modelo'].unique()
    )

# Sección de gráficas
if pagina_seleccionada == "📈 Visualización de Gráficas":
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

# Filtrar datos solo si estamos en sección de análisis
if pagina_seleccionada == "📊 Análisis y Métricas":
    df_filtrado = df[df['Modelo'].isin(modelos_seleccionados)]
else:
    df_filtrado = df[df['Modelo'].isin(df['Modelo'].unique())]

# Si se selecciona visualización de gráficas, mostrar el visor
if pagina_seleccionada == "📈 Visualización de Gráficas":
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

        # Se hace una lista con tal de que se muestre el modelo con el que se entrenó el modelo cada vez que se seleccione una zona de la lista desplegable
        zonas_y_modelos = {
            "001_ZW Parque Ingenio": "Random Forest",
            "002_ZW Canchas Panamericanas": "SVR",
            "003_ZW Parque del Perro": "SVR",
            "004_ZW Parque San Nicolas": "Regresion Lineal",
            "005_ZW Parque Barrio Obrero": "Random Forest",
            "007_ZW Parque Pizamos": "Perceptron",
            "008_ZW Parque Alfonso Lopez": "SVR",
            "010_ZW Parque Antonio Nariño": "Regresion Lineal",
            "011_ZW Parque Santa Rosa Poblado": "Random Forest",
            "012_ZW Polideportivo Los Naranjos": "Random Forest",
            "013_ZW Parque Llano Verde": "Random Forest",
            "015_ZW Centro Cultural Comuna 13": "Random Forest",
            "016_ZW Conjunto Habitacional Ramali": "Random Forest",
            "019_ZW Parque Skate Board": "SVR",
            "021_ZW Parque Alfonso Barberena": "Regresion Lineal",
            "022_ZW Polideportivo Los Farallones": "Random Forest",
            "023_ZW Parque Los Guerreros": "SVR",
            "024_ZW Museo La Tertulia": "Random Forest",
            "025_ZW Parque San Marino": "Random Forest",
            "026_ZW Parque La Flora": "Regresion Lineal",
            "027_ZW Parque Alfonso Bonilla Aragón": "Random Forest",
            "028_ZW Parque Yo Amo a Siloé": "Regresion Lineal",
            "029_ZW Centro Cultural Vista Hermosa": "SVR",
            "030_ZW Parque Mutis": "Regresion Lineal",
            "031_ZW Cancha Los Azules": "Regresion Lineal",
            "032_ZW Parque La Orqueta": "Random Forest",
            "034_ZW Parque Villa Colombia": "Random Forest",
            "035_ZW Parque Colseguros": "Random Forest",
            "036_ZW Parque India Elena": "Regresion Lineal",
            "037_ZW Parque Tequendama": "Perceptron",
            "038_ZW Parque Sector Amarillo Skate Park": "Random Forest",
            "039_ZW Biblioteca Daniel Guillard": "Random Forest",
            "040_ZW Polideportivo Torres Comfandi": "Random Forest",
            "041_ZW Parque Junin": "Random Forest",
            "042_ZW Parque Mariano Ramos": "SVR",
            "043_ZW Polideportivo Ricardo Balcazar": "Regresion Lineal",
            "044_ZW Polideportivo San Benito": "Regresion Lineal",
            "045_ZW Parque Santa Anita": "SVR",
            "046_ZW Sebastian Belalcazar": "Random Forest",
            "047_ZW Parque del Mico": "SVR",
            "048_ZW Parque Cien Palos": "Regresion Lineal",
            "049_ZW Cerro 3 Cruces": "SVR",
            "050_ZW Parque Calima": "SVR",
            "051_ZW Parque La Merced": "Random Forest",
            "052_ZW Puente de Colores": "Random Forest",
            "053_ZW Polideportivo Laureano Gomez": "Random Forest",
            "054_ZW El Diamante": "Random Forest",
            "055_ZW Polideportivo Petecuy": "Random Forest",
            "056_ZW Comuna 16": "Random Forest",
            "057_La Castilla": "SVR",
            "058_La Elvira": "Perceptron",
            "059_El Saladito": "SVR",
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
        st.stop()

# ============================================================================
# SECCIÓN: PREDICCIÓN INTERACTIVA (ÁREA PRINCIPAL)
# ============================================================================
if pagina_seleccionada == "🔮 Predicción Interactiva":
    # Mensaje de bienvenida por defecto
    if not boton_prediccion:
        st.markdown("""
        # Bienvenido a: Predicción Interactiva de Tráfico WiFi de la Ciudad de Cali
        
        👈 **Utilice el panel lateral izquierdo para seleccionar la zona WiFi y los demás parámetros requeridos, 
        luego presione 'Generar Predicción' para ver los resultados.**
        """)
    else:
        # Mostrar resultados cuando se presiona el botón
        st.header("🔮 Resultados de la Predicción")
        
        with st.spinner("Procesando predicción..."):
            try:
                # Se normalizan los nombres para que no dé error por las ñ's y tildes
                if zona_seleccionada_pred[:3] == "010":
                    zona_seleccionada_pred = "010_ZW Parque Antonio Nariño"
                    #zona_seleccionada_pred = "010_ZW Parque Antonio NariÃ±o"
                if zona_seleccionada_pred[:3] == "027":
                    zona_seleccionada_pred = "027_ZW Parque Alfonso Bonilla Aragón"
                if zona_seleccionada_pred[:3] == "028":
                    zona_seleccionada_pred = "028_ZW Parque Yo Amo a Siloé"

                # Se carga el modelo guardado de la zona seleccionada
                modelo_completo = joblib.load(open(MODELOS_DIR / f"{zona_seleccionada_pred}.joblib", 'rb'))

                # Se carga dataset de la zona seleccionada (esto se hace para incluír meses pasados dentro de la gráfica de predicción)
                df = pd.read_csv(ZONAS_DIR / f"{zona_seleccionada_pred}.csv")
                
                modelo_entrenado = modelo_completo["metadata"]["model_type"]
                ventana_datos = modelo_completo['ventana_datos']

                nombre_zona = zona_seleccionada_pred[6:]

                st.write(f"Mejor modelo desempeñado para {nombre_zona}: {modelo_entrenado}")
                st.write("Variables de entrada para la predicción:")

                last_window = modelo_completo["forecaster"].last_window_
                ultima_fecha = last_window.index.tolist()[-1]
                
                # Crear rango de fechas futuras
                fechas_futuras = pd.date_range(
                    start=ultima_fecha + pd.Timedelta(days=7),
                    periods=1,
                    freq='D'
                )

                # Crear DataFrame para exógenas futuras
                exog_futura = pd.DataFrame(index=fechas_futuras)
                exog_futura['DIA_SEMANA'] = fechas_futuras.dayofweek
                exog_futura['LABORAL'] = (exog_futura['DIA_SEMANA'] < 5).astype(int)
                exog_futura['FIN_DE_SEMANA'] = (exog_futura['DIA_SEMANA'] >= 5).astype(int)
                exog_futura['FESTIVO'] = 0
                exog_futura["NUMERO_CONEXIONES"] = numero_conexiones_1
                exog_futura["PORCENTAJE_USO"] = porcentaje_uso_1

                # Seguir con los siguientes dias....

                st.table(exog_futura)

                st.write("Variables de entrada de 10 días atrás de la zona:")

                st.table(modelo_completo["ventana_datos"].tail(10))

                exog_variables = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO', 'NUMERO_CONEXIONES']
                exog_variables_scaled = ['DIA_SEMANA', 'LABORAL', 'FIN_DE_SEMANA', 'FESTIVO', 'PORCENTAJE_USO_scaled', 'NUMERO_CONEXIONES_scaled']

                prediccion_final = 0

                dias_a_predecir = 1 # NOTA: "dias_a_predecir" debe ser una variable enviada desde el frontend
                df_con_lags, df_junto, exog_variables_lags = crear_lags_exogenos(exog_futura, ventana_datos, 10, dias_a_predecir)

                if modelo_entrenado == "Random Forest":

                    prediccion_1_dia = modelo_completo["forecaster"].predict(
                        steps=1,
                        #exog=exog_futura[exog_variables]
                        exog=df_con_lags[exog_variables_lags]
                    )

                    prediccion_final = prediccion_1_dia

                else: #(para SVR, Regresion Lineal o Perceptron)
                    
                    # Primero escalamos
                    scaler_usage = modelo_completo['scalers']['scaler_usage']
                    scaler_conexiones = modelo_completo['scalers']['scaler_conexiones']
                    scaler_porcentaje = modelo_completo['scalers']['scaler_porcentaje']

                    # Se escalan las exogenas futuras aplicando los escaladores de train (Se usa "transform", no "fit_transform")
                    #exog_futura['NUMERO_CONEXIONES_scaled'] = scaler_conexiones.transform(exog_futura[['NUMERO_CONEXIONES']])
                    #exog_futura['PORCENTAJE_USO_scaled'] = scaler_porcentaje.transform(exog_futura[['PORCENTAJE_USO']])

                    #prediccion_1_dia_scaled = modelo_completo["forecaster"].predict(
                    #    steps=1,
                    #    exog=exog_futura[exog_variables_scaled]
                    #)

                    # Identificamos las columnas que empiezan con el prefijo
                    USAGE_KB_to_scale = df_con_lags.filter(like='USAGE_KB').columns
                    NUMERO_CONEXIONES_to_scale = df_con_lags.filter(like='NUMERO_CONEXIONES').columns
                    PORCENTAJE_USO_to_scale = df_con_lags.filter(like='PORCENTAJE_USO').columns

                    # Se escala df_con_lags aplicando los escaladores de train (Se usa "transform", no "fit_transform", este último se debe usar en la data a predecir)
                    df_con_lags[NUMERO_CONEXIONES_to_scale] = scaler_conexiones.transform(df_con_lags[NUMERO_CONEXIONES_to_scale])
                    df_con_lags[PORCENTAJE_USO_to_scale] = scaler_porcentaje.transform(df_con_lags[PORCENTAJE_USO_to_scale])

                    # Predecir días con variables exógenas
                    prediccion_1_dia_scaled = modelo_completo["forecaster"].predict(
                        steps=1,
                        exog=df_con_lags[exog_variables_lags]
                    )

                    # Despues desescalamos
                    # Desescalado de la prediccion con los hiperparámetros encontrados en la grilla:
                    prediccion_desescalada = pd.Series(
                        scaler_usage.inverse_transform(prediccion_1_dia_scaled.values.reshape(-1, 1)).flatten(),
                        index=prediccion_1_dia_scaled.index
                    )

                    prediccion_final = prediccion_desescalada

                fecha_predicha = prediccion_final.index[0]
                month = fecha_predicha.strftime("%B")
                dia = fecha_predicha.strftime("%d")
                anio = fecha_predicha.strftime("%Y")
                if month == "January":
                    mes = "Enero"
                elif month == "February":
                    mes = "Febrero"
                elif month == "March":
                    mes = "Marzo"
                elif month == "April":
                    mes = "Abril"
                elif month == "May":
                    mes = "Mayo"
                elif month == "June":
                    mes = "Junio"
                elif month == "July":
                    mes = "Julio"
                elif month == "August":
                    mes = "Agosto"
                elif month == "September":
                    mes = "Septiembre"
                elif month == "October":
                    mes = "Octubre"
                elif month == "November":
                    mes = "Noviembre"
                else:
                    mes = "Diciembre"
                

                # Obtener métricas de la zona seleccionada
                #zona_metricas = df_mejores_zona[df_mejores_zona['ZONA'].str.contains(
                #    zona_seleccionada_pred.split('_')[-1] if '_' in zona_seleccionada_pred else zona_seleccionada_pred, 
                #    case=False, na=False
                #)]
                
                zona_metricas = df_mejores_zona.query(f"ZONA.str.contains('{zona_seleccionada_pred[:3]}')")

                #st.write(zona_metricas)
                #st.write(result_query_like)
                
                # Calcular predicción simulada (aquí iría la predicción real)
                #trafico_predicho = numero_conexiones * porcentaje_uso / 100 * 10  # Simulación simple
                
                # Crear tabla de resultados
                resultados_prediccion = pd.DataFrame({
                    'Zona': [zona_seleccionada_pred],
                    'Número_Conexiones': [numero_conexiones],
                    'Porcentaje_Uso_%': [porcentaje_uso],
                    'Tráfico_Predicho_KB': [prediccion_final[0]]
                })
                
                # Mostrar tabla de resultados
                st.subheader("📊 Resultados")

                st.write(f"La predicción final para esta zona para {mes} {dia} de {anio} es:")
                st.write(f"{int(prediccion_final[0])} KB")

                st.table(prediccion_final)

                st.dataframe(
                    resultados_prediccion,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Métricas de la zona
                if not zona_metricas.empty:
                    st.subheader("📈 Métricas Históricas de la Zona")
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    
                    with col_met1:
                        st.metric("MAPE Histórico", f"{zona_metricas.iloc[0]['MAPE_NUM']:.2f}%")
                    
                    with col_met2:
                        st.metric("Mejor Modelo", zona_metricas.iloc[0]['MEJOR_MODELO_ENCONTRADO'])
                    
                    with col_met3:
                        st.metric("Tráfico Predicho (KB)", f"{prediccion_final[0]:.2f}")
                    
                    with col_met4:
                        st.metric("Zona", zona_seleccionada_pred)
                
                # Gráfico de tendencia (simulado)
                st.subheader("📉 Predicción de Tráfico")
                st.write("(Se muestran los dos últimos meses junto con el día predicho)")

                plt.style.use('seaborn-v0_8-dark')
                plt.figure(figsize=(25, 4))
                plt.plot(df['USAGE_KB'], label="Train", linewidth=2)
                plt.plot(prediccion_final[0], label="Predicho", linewidth=2)
                plt.title(f"{modelo_entrenado} - {nombre_zona}")
                plt.xlabel("Índice temporal")
                plt.ylabel("Tráfico (kB)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.close()

                
                fig_tendencia = px.line(
                    df.tail(60),
                    x='FECHA_CONEXION',
                    y='USAGE_KB',
                    title=f'Predicción de Tráfico - {zona_seleccionada_pred}',
                    labels={'Tráfico_KB': 'Tráfico (KB)', 'Fecha': 'Fecha'}
                )

                # Marcar el punto de predicción
                fig_tendencia.add_scatter(
                    x=[prediccion_final.index[0]],
                    y=[prediccion_final[0]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Predicción'
                )
                
                st.plotly_chart(fig_tendencia, use_container_width=True)

                mape_historico = int(zona_metricas.iloc[0]['MAPE_NUM'])

                if mape_historico <= 20:
                    descrpcion_mape = "Un valor MAPE menor o igual al 20% (80% de exactitud) nos indica que la predicción fue muy buena si se compara con el estándar de predicciones en Ciencia de Datos"
                if 30 >= mape_historico > 20:
                    descrpcion_mape = "Un valor MAPE mayor a 20% pero menor o igual al 30% (70% de exactitud) nos indica que la predicción fue buena si se compara con el estándar de predicciones en Ciencia de Datos"
                if 50 >= mape_historico > 30:
                    descrpcion_mape = "Un valor MAPE mayor a 30% pero menor o igual al 50% (50% de exactitud) nos indica que la predicción fue regular si se compara con el estándar de predicciones en Ciencia de Datos"
                if mape_historico > 50:
                    descrpcion_mape = "Un valor MAPE mayor a 50% (menos del 50% de exactitud) nos indica que la predicción fue mala. Esto no es una falla en la predicción sino una característica de la serie temporal de la zona que hace de sus picos bruscos (de 0 a tráficos altos) un comportamiento muy dificil para predecir"
                
                # Información adicional
                with st.expander("ℹ️ Información sobre la Predicción"):
                    st.markdown(f"""
                    **Zona Seleccionada:** {zona_seleccionada_pred}
                    
                    **Parámetros Ingresados:**
                    Día 1:
                    - Número de Conexiones: {numero_conexiones_1}
                    - Porcentaje de Uso: {porcentaje_uso_1}%

                    Día 2:
                    - Número de Conexiones: {numero_conexiones_2}
                    - Porcentaje de Uso: {porcentaje_uso_2}%

                    Día 3:
                    - Número de Conexiones: {numero_conexiones_3}
                    - Porcentaje de Uso: {porcentaje_uso_3}%

                    Día 4:
                    - Número de Conexiones: {numero_conexiones_4}
                    - Porcentaje de Uso: {porcentaje_uso_4}%

                    Día 5:
                    - Número de Conexiones: {numero_conexiones_5}
                    - Porcentaje de Uso: {porcentaje_uso_5}%

                    Día 6:
                    - Número de Conexiones: {numero_conexiones_6}
                    - Porcentaje de Uso: {porcentaje_uso_6}%

                    Día 7:
                    - Número de Conexiones: {numero_conexiones_7}
                    - Porcentaje de Uso: {porcentaje_uso_7}%
                    
                    **Predicción:**
                    - Tráfico Estimado: {prediccion_final[0]:.2f} KB
                    
                    **Nota:** Esta es una predicción basada en los parámetros ingresados.
                    
                    El sistema utiliza modelos de machine learning
                    entrenados con datos históricos.
                    
                    El valor del MAPE Histórico ({zona_metricas.iloc[0]['MAPE_NUM']:.2f}%) es el valor del error porcentual absoluto medio.
                    Indica porcentualmente en promedio cuán lejos estuvieron las pruebas previas de predicción (proceso que se hace comparando datos de prueba con datos de entrenamiento del modelo de Machine Learning).
                    
                    {descrpcion_mape}
                    """)
                
            except Exception as e:
                st.error(f"Error al procesar la predicción: {e}")
                st.info("Por favor, verifique que todos los datos estén correctamente ingresados.")
    
    st.stop()

# ============================================================================
# SECCIÓN A: COMPARACIÓN DE MODELOS (solo se muestra en Análisis y Métricas)
# ============================================================================
if pagina_seleccionada == "📊 Análisis y Métricas":
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
    textinfo='value+percent+label',
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

    # Footer
    st.markdown("---")
    st.caption("Dashboard Proyecto Predicción Zonas WiFi de Cali")



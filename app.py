# ===========================================================================
# PROJECT: 
# MODULE: 
# ---------------------------------------------------------------------------
# DEVELOPED AND IMPLEMENTED BY: Juan Carlos Pantoja
# (ML Modeling, Scripting, and Application Development)
#
# TECHNICAL REVIEW: 
# PROJECT SUPERVISION: 
# INSTITUTION: Universidade do Minho
# ---------------------------------------------------------------------------
# SCIENTIFIC DISCLAIMER:
# These models are part of an exploratory study. This framework is designed
# for incremental refinement as more sensor data becomes available. Results
# represent preliminary trends and should be interpreted within the specific
# research scope of the Task 4 Performance Evaluation report.
# ===========================================================================

# import os
import io
import pandas as pd
import streamlit as st
from src.predictor import UniversalPredictor

# ===========================================================================
# CONFIGURACIÓN MAESTRA DE MODELOS
# ===========================================================================
MODELS_CONFIG = {

    "structural_classifier": {
        "type": "CatBoost",
        "features": ['M', 'gwt', 'pga', 'lpi', 'liq_man_cat'],  
        "cat_features": []
    },
    "geotechnical_classifier": {
        "type": "CatBoost",
        "features": ['M', 'gwt', 'pga', 'lpi'],
        "cat_features": []
    },

    # "2": {
    #     "type": "Tensorflow",
    #     "features": ['dummy1', 'dummy2']
    # },
    # "8": {
    #     "type": "XGBoost",
    #     "features": ['dummy1', 'dummy2']
    # }
}

# Inicializar componentes
MODELS_DIR = "models"  # Cerciórate de colocar tus archivos model_*.cbm, model_*.keras aquí
predictor = UniversalPredictor(models_folder=MODELS_DIR)

# Configuración de página de Streamlit
st.set_page_config(page_title="Framework Universal de Predicción", layout="wide", page_icon="🏗️")

st.title("🏗️ Liquefaction Severity Manifestation Probability: LiSeM-Pro")
st.caption("Developed by Juan Carlos Pantoja")

# Sidebar para control
st.sidebar.header("⚙️ Configuración")
model_id = st.sidebar.selectbox("Seleccione el Modelo para Analizar:", list(MODELS_CONFIG.keys()))

model_info = MODELS_CONFIG[model_id]
st.sidebar.info(f"**Tipo de Framework:** {model_info['type']}")

# --- PASO 1: GENERACIÓN AUTOMÁTICA DE PLANTILLAS ---
st.header("1. Descargar Esquema del Modelo")
st.write(f"Descargue la plantilla de Excel vacía estructurada con los requerimientos exactos de columnas para el modelo **{model_id}**.")

# Creamos el dataframe vacío con los headers del modelo
df_template = pd.DataFrame(columns=model_info["features"])

# Escribir a memoria (BytesIO) para evitar conflictos de escritura en el servidor
template_buffer = io.BytesIO()
with pd.ExcelWriter(template_buffer, engine="xlsxwriter") as writer:
    df_template.to_excel(writer, index=False, sheet_name=str(model_id))
template_buffer.seek(0)

st.download_button(
    label="📥 Descargar Plantilla Excel",
    data=template_buffer,
    file_name=f"plantilla_modelo_{model_id}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")

# --- PASO 2: CARGA DE DATOS Y PREDICCIÓN ---
st.header("2. Cargar Datos y Procesar Inferencias")
uploaded_file = st.file_uploader("Suba la plantilla Excel completada:", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Cargar datos subidos por el usuario
        df_input = pd.read_excel(uploaded_file)
        st.success("✅ Archivo cargado exitosamente.")
        
        # Validación de Columnas
        missing_cols = [col for col in model_info["features"] if col not in df_input.columns]
        
        if missing_cols:
            st.error(f"🚨 Error: Al archivo le faltan las siguientes columnas obligatorias: {missing_cols}")
        else:
            st.subheader("👀 Vista Previa de Datos de Entrada")
            st.dataframe(df_input.head(5))
            
            if st.button("🚀 Ejecutar Predicciones por Lote"):
                with st.spinner("Procesando datos a través de UniversalPredictor..."):
                    
                    # Extraer solo las columnas ordenadas que el modelo espera
                    X_data = df_input[model_info["features"]]
                    
                    # Llamar al predictor universal pasándole parámetros opcionales de categóricas si es CatBoost
                    cat_f = model_info.get("cat_features", None)
                    results = predictor.predict(
                        model_name=model_id, 
                        model_type=model_info["type"], 
                        data=X_data,
                        cat_features=cat_f
                    )
                    
                    # Adjuntar resultados al DataFrame
                    df_output = df_input.copy()
                    df_output["Model_Prediction"] = results
                    
                    st.success("🎉 ¡Procesamiento completado con éxito!")
                    st.subheader("📊 Muestra de Resultados con Predicciones")
                    st.dataframe(df_output.head(5))
                    
                    # Crear el archivo Excel de descarga final en memoria
                    output_buffer = io.BytesIO()
                    with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
                        df_output.to_excel(writer, index=False, sheet_name="Resultados_Prediccion")
                    output_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Descargar Resultados (.XLSX)",
                        data=output_buffer,
                        file_name=f"resultados_modelo_{model_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
    except Exception as e:
        st.error(f"💥 Error crítico en el pipeline de ejecución: {e}")
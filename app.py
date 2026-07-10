# ===========================================================================
# PROJECT: Seismic vulnerability assessment model for urban centres prone to
#          liquefaction: Insights into a multi-hazard approach
# APPLICATION: Surface Liquefaction manifestation severity and probability classifier
# MODULE: Liquefaction Vulnerability Module
# ---------------------------------------------------------------------------
# DEVELOPED AND IMPLEMENTED BY: Juan Carlos Pantoja
# (ML Modeling, Scripting, and Application Development)
#
# SCIENTIFIC METHODOLOGY:
# Alejandro Cruz Escobar
# Doctoral Researcher, Civil Engineering
# Universidade do Minho
#
# PROJECT SUPERVISION:
# Professor Paulo B. Lourenço
# Doctor Shaghayegh Karimzadeh
# Doctor Nicola Chieffo
# INSTITUTION: Universidade do Minho
# ---------------------------------------------------------------------------
# SCIENTIFIC DISCLAIMER:
# This application is part of an exploratory doctoral research framework for
# assessing liquefaction-related vulnerability in urban areas. The implemented
# models are intended for research, methodological development, and preliminary
# screening purposes.
#
# The results should not be interpreted as definitive site-specific predictions
# of liquefaction damage or as a substitute for detailed geotechnical,
# structural, or engineering investigations. Model outputs depend on the quality,
# representativeness, and range of the input data used for training and
# calibration.
#
# The framework is designed for incremental refinement as additional
# post-earthquake liquefaction observations and building damage data become
# available. Therefore, the results should be interpreted within the scientific
# scope of Chapter 3, "Methodological Framework: Vulnerability Modules and
# Unified multi-hazard integration", of the doctoral thesis.
# ===========================================================================

import io
import pandas as pd
import streamlit as st
from src.predictor import UniversalPredictor

# ===========================================================================
# MASTER MODEL CONFIGURATION
# ===========================================================================
MODELS_CONFIG = {
    "geotechnical_classifier": {
        "type": "CatBoost",
        "features": ['M', 'gwt', 'pga', 'lpi'],
        "cat_features": [],
        "class_mapping": {
            0: "N/Mi",
            1: "Mo",
            2: "S",
        }
    },
    "structural_classifier": {
        "type": "CatBoost",
        "features": ['M', 'gwt', 'pga', 'lpi', 'liq_man_cat'],
        "cat_features": [],
        "class_mapping": {
            0: "No damage",
            1: "Damage",
        }
    }
}

MODELS_DIR = "models"
predictor = UniversalPredictor(models_folder=MODELS_DIR)

# Streamlit Page Setup
st.set_page_config(page_title="Universal Prediction Framework", layout="wide", page_icon="🏗️")

st.title("🏗️ Liquefaction Severity Manifestation Probability: LiSeM-Pro")
st.caption("Developed by **Juan Carlos Pantoja**")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
model_id = st.sidebar.selectbox("Select Model to Analyze:", list(MODELS_CONFIG.keys()))

model_info = MODELS_CONFIG[model_id]
st.sidebar.info(f"**Framework Type:** {model_info['type']}")

# Display Target Classes in Sidebar
with st.sidebar.expander("🏷️ Configured Classes"):
    for code, label in model_info["class_mapping"].items():
        st.write(f"**{code}**: {label}")

with st.sidebar.expander("ℹ️ Scientific Disclaimer", expanded=True):
    st.markdown(
        """
        **PROJECT:**  
        Seismic vulnerability assessment model for urban centres prone to liquefaction: Insights into a multi-hazard approach.

        **APPLICATION:**  
        Surface Liquefaction manifestation severity and probability classifier.

        **MODULE:**  
        Liquefaction Vulnerability Module.

        ---

        **DEVELOPED AND IMPLEMENTED BY:**  
        * **Juan Carlos Pantoja** (ML Modeling, Scripting, and Application Development)

        **SCIENTIFIC METHODOLOGY:**  
        * **Alejandro Cruz Escobar** (Doctoral Researcher, Civil Engineering)

        **PROJECT SUPERVISION:**  
        * **Professor Paulo B. Lourenço**  
        * **Doctor Shaghayegh Karimzadeh**  
        * **Doctor Nicola Chieffo**  

        **INSTITUTION:**  
        Universidade do Minho  

        ---

        ⚠️ **Educational & Research Tool Only**  
        This app is part of an exploratory doctoral framework at *Universidade do Minho* for screening liquefaction-related vulnerabilities.

        Outputs represent preliminary trends and **must not** be used as a substitute for site-specific geotechnical, structural, or professional engineering investigations.
        """
    )

# --- STEP 1: AUTOMATIC TEMPLATE GENERATION ---
st.header("1. Download Model Template")
st.write(
    f"Download the empty Excel template structured with the exact column schema required for the **{model_id}** model.")

# Create empty DataFrame with required feature headers
df_template = pd.DataFrame(columns=model_info["features"])

# Write to memory buffer (BytesIO)
template_buffer = io.BytesIO()
with pd.ExcelWriter(template_buffer, engine="xlsxwriter") as writer:
    df_template.to_excel(writer, index=False, sheet_name=str(model_id))
template_buffer.seek(0)

st.download_button(
    label="📥 Download Excel Template",
    data=template_buffer,
    file_name=f"template_model_{model_id}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")

# --- STEP 2: DATA UPLOAD AND INFERENCE ---
st.header("2. Upload Data and Run Predictions")
uploaded_file = st.file_uploader("Upload your completed Excel template here:", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load user-submitted file
        df_input = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully.")

        # Feature Column Validation
        missing_cols = [col for col in model_info["features"] if col not in df_input.columns]

        if missing_cols:
            st.error(f"🚨 Error: The uploaded file is missing the following required columns: {missing_cols}")
        else:
            st.subheader("👀 Input Data Preview")
            st.dataframe(df_input.head(5))

            if st.button("🚀 Run Batch Predictions"):
                with st.spinner("Processing data through UniversalPredictor..."):

                    # Slice out only the features required by the model in the precise order
                    X_data = df_input[model_info["features"]]
                    cat_f = model_info.get("cat_features", None)

                    # Extract discrete predictions and probability DataFrame
                    results, df_probs = predictor.predict(
                        model_name=model_id,
                        model_type=model_info["type"],
                        data=X_data,
                        cat_features=cat_f
                    )

                    # Build output DataFrame
                    df_output = df_input.copy()

                    # 1. Store original numerical code
                    df_output["Model_Prediction_Code"] = results

                    # 2. Map indices to structured text labels
                    mapping = model_info["class_mapping"]
                    df_output["Model_Prediction_Label"] = pd.Series(results).map(mapping).values

                    # 3. Rename probability columns and concatenate
                    if df_probs is not None:
                        new_col_names = {}
                        for col in df_probs.columns:
                            idx = int(col.split("_")[-1])
                            if idx in mapping:
                                new_col_names[col] = f"Prob_{mapping[idx]}"
                            else:
                                new_col_names[col] = col

                        df_probs = df_probs.rename(columns=new_col_names)
                        df_output = pd.concat([df_output.reset_index(drop=True), df_probs.reset_index(drop=True)],
                                              axis=1)

                    st.success("🎉 Processing completed successfully!")
                    st.subheader("📊 Output Sample (Features + Predictions)")
                    st.dataframe(df_output.head(5))

                    # Build exportable Excel binary file in memory
                    output_buffer = io.BytesIO()
                    with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
                        df_output.to_excel(writer, index=False, sheet_name="Prediction_Results")
                    output_buffer.seek(0)

                    st.download_button(
                        label="💾 Download Results (.XLSX)",
                        data=output_buffer,
                        file_name=f"results_model_{model_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    except Exception as e:
        st.error(f"💥 Critical error inside the execution pipeline: {e}")
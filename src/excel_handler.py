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

import pandas as pd
import os

def create_excel_template(file_path, models_features):
    """
    Creates an Excel file where each sheet represents a model.
    Columns are based on the required features for that specific model.
    """
    try:
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            for model_name, info in models_features.items():
                # Create an empty DataFrame with the required headers
                features = info["features"]
                df_template = pd.DataFrame(columns=features)
                df_template.to_excel(writer, sheet_name=model_name, index=False)
        print(f"Success: Template created at {file_path}")
    except Exception as e:
        print(f"Error creating Excel template: {e}")

def load_data_from_excel(file_path, model_name):
    """
    Reads a specific sheet from the Excel file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Read the specific sheet for the model
    data = pd.read_excel(file_path, sheet_name=model_name)
    
    return data
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
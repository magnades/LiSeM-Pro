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

import os
import joblib
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
from catboost import CatBoostClassifier, Pool

class UniversalPredictor:
    def __init__(self, models_folder="models"):
        self.models_folder = models_folder
        self.cache = {}

    def _load_assets(self, model_name, model_type):
        match model_type:
            case "XGBoost":
                skl_path = os.path.join(self.models_folder, f"model_{model_name}.pkl")
                if os.path.exists(skl_path):
                    print(f"--- Loading SKLearn/XGBoost Pipeline for Model: {model_name} ---")
                    return joblib.load(skl_path), "sklearn"
                raise FileNotFoundError(f"No pkl found for {model_name}")

            # case "Tensorflow":
            #     tf_model_path = os.path.join(self.models_folder, f"model_{model_name}.keras")
            #     tf_prep_path = os.path.join(self.models_folder, f"preprocessor_{model_name}.pkl")
            #     if os.path.exists(tf_model_path) and os.path.exists(tf_prep_path):
            #         print(f"--- Loading TensorFlow MLP for Model: {model_name} ---")
            #         model = load_model(tf_model_path)
            #         preprocessor = joblib.load(tf_prep_path)
            #         return {"model": model, "preprocessor": preprocessor}, "tensorflow"
            #     raise FileNotFoundError(f"Missing keras/preprocessor files for {model_name}")

            case "CatBoost": 
                cb_path = os.path.join(self.models_folder, f"model_{model_name}.cbm")
                if os.path.exists(cb_path):
                    print(f"--- Loading CatBoost Model: {model_name} ---")
                    model = CatBoostClassifier()
                    model.load_model(cb_path)
                    return model, "catboost"
                raise FileNotFoundError(f"No .cbm file found for {model_name}")

            case _:
                raise ValueError(f"Unsupported framework type: '{model_type}'")

    def predict(self, model_name, model_type, data, cat_features=None):
        """
        Executes prediction regardless of the underlying framework.
        """
        if model_name not in self.cache:
            self.cache[model_name] = self._load_assets(model_name, model_type)
        
        model_obj, framework = self.cache[model_name]

        if framework == "sklearn":
            return model_obj.predict(data)
        
        # elif framework == "tensorflow":
        #     processed_data = model_obj["preprocessor"].transform(data)
        #     predictions = model_obj["model"].predict(processed_data, verbose=0)
        #     return predictions.flatten()
            
        elif framework == "catboost":  
            # Asegurar tratamiento de nulos básicos en categóricas antes del Pool
            if cat_features:
                data_cp = data.copy()
                for col in cat_features:
                    if col in data_cp.columns:
                        data_cp[col] = data_cp[col].astype(str).fillna("Missing")
                pool = Pool(data_cp, cat_features=cat_features)
            else:
                pool = Pool(data)
            
            return model_obj.predict(pool).astype(int).flatten()
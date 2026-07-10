# ===========================================================================
# PROJECT: Liquefaction Severity Manifestation Probability: LiSeM-Pro
# MODULE: Machine Learning Prediction Framework
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
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool  # <-- Dependencia nativa para tus nuevos clasificadores


class UniversalPredictor:
    def __init__(self, models_folder="models"):
        """
        Handles different ML frameworks (XGBoost, CatBoost).
        """
        self.models_folder = models_folder
        self.cache = {}

    def _load_assets(self, model_name, model_type):
        """
        Internal logic to identify and load model files.
        """
        match model_type:
            case "XGBoost":
                # Estrategia A: Pipeline clásico Scikit-Learn/XGBoost (.pkl)
                skl_path = os.path.join(self.models_folder, f"model_{model_name}.pkl")
                if os.path.exists(skl_path):
                    print(f"--- Loading SKLearn/XGBoost Pipeline for Model: {model_name} ---")
                    return joblib.load(skl_path), "sklearn"
                raise FileNotFoundError(f"Missing .pkl asset for model '{model_name}' in {self.models_folder}")

            case "CatBoost":
                # Estrategia C: Modelos nativos de CatBoost (.cbm)
                cb_path = os.path.join(self.models_folder, f"model_{model_name}.cbm")
                if os.path.exists(cb_path):
                    print(f"--- Loading CatBoost Model: {model_name} ---")
                    model = CatBoostClassifier()
                    model.load_model(cb_path)
                    return model, "catboost"
                raise FileNotFoundError(f"Missing .cbm asset for CatBoost model '{model_name}' in {self.models_folder}")

            case _:
                raise ValueError(f"Unsupported framework type: '{model_type}'")

    def predict(self, model_name, model_type, data, cat_features=None):
        """
        Executes prediction regardless of the underlying framework.
        Returns a tuple: (predictions, probabilities_dataframe)
        """
        if model_name not in self.cache:
            self.cache[model_name] = self._load_assets(model_name, model_type)

        model_obj, framework = self.cache[model_name]

        if framework == "sklearn":
            # Los pipelines de SKLearn manejan internamente codificación y escalado
            return model_obj.predict(data), None

        elif framework == "catboost":
            # Preparar un subset de datos limpio y tipar categóricas si existen
            data_cp = data.copy()
            if cat_features:
                for col in cat_features:
                    if col in data_cp.columns:
                        data_cp[col] = data_cp[col].astype(str).fillna("Missing")
                pool = Pool(data_cp, cat_features=cat_features)
            else:
                pool = Pool(data_cp)

            # 1. Obtener predicciones discretas de clases (0, 1, 2...)
            preds = model_obj.predict(pool).astype(int).flatten()

            # 2. Extraer probabilidades numéricas por fila para cada clase disponible
            prob_matrix = model_obj.predict_proba(pool)

            # 3. Empaquetar matriz en un DataFrame usando los nombres nativos de las clases internas
            # (Si las clases en el entrenamiento fueron numéricas [0, 1, 2], genera 'Prob_Clase_0', etc.)
            class_indices = model_obj.classes_
            df_probs = pd.DataFrame(
                prob_matrix,
                columns=[f"Prob_Clase_{idx}" for idx in class_indices]
            )

            return preds, df_probs
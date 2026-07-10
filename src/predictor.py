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

import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool


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

            class_indices = model_obj.classes_
            df_probs = pd.DataFrame(
                prob_matrix,
                columns=[f"Prob_Clase_{idx}" for idx in class_indices]
            )

            return preds, df_probs
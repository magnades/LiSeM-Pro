# Machine-Learning-Based Framework for SLM and LIBD Classification (LiSeM-Pro)

## 1. Project Overview

This tool implements Stage 2 of the framework developed in the study "Urban-scale assessment of liquefaction-induced building damage using fragility-informed indices and machine learning". Developed at the Universidade do Minho in July 2026, the application supports the evaluation of surface liquefaction manifestations (SLMs) and liquefaction-induced building damage (LIBD). It utilizes supervised machine-learning models trained on harmonised post-earthquake observations from New Zealand, Italy, and Chile.

The main purpose of the tool is to provide a reproducible implementation of the trained machine-learning models and facilitate the application of the framework for urban-scale liquefaction-vulnerability screening. The predictions support the identification and prioritisation of sites potentially affected by moderate or severe surface manifestations and liquefaction-induced building damage.

## 2. Features and Models

The framework includes two CatBoost classification models:

### Model 3: Surface Liquefaction Manifestation (SLM) Classification

*   **Target:** Predicts SLM severity.
*   **Classification Classes:**
    *   0: No or minor manifestation (N/Mi).
    *   1: Moderate manifestation (Mo).
    *   2: Severe manifestation (S).
*   **Predictors Used:** Earthquake magnitude (M), Peak ground acceleration (PGA), Groundwater-table depth (GWT), and Liquefaction potential index (LPI).
*   **Data Background:** Developed using 13,328 harmonised observations and evaluated on a 20% hold-out test set.

### Model 4: Liquefaction-Induced Building Damage (LIBD) Classification

*   **Target:** Predicts binary building-damage occurrence.
*   **Classification Classes:** 
    *   0: No damage. 
    *   1: Damage.
*   **Predictors Used:** Earthquake magnitude (M), Peak ground acceleration (PGA), Groundwater-table depth (GWT), Liquefaction potential index (LPI), and Observed SLM severity class.
*   **Data Background:** Developed using a smaller Italy–Chile subset containing 130 observations and evaluated on a 26-site hold-out test set.
*   **Usage Note:** Because Model 4 uses observed SLM severity as an input, its current implementation is primarily suitable for post-earthquake or conditional damage assessment. For prospective applications, the observed SLM input must be replaced with SLM probabilities or classifications obtained from Model 3 or another independently validated manifestation model.

## 3. File Structure & Application Usage

### Directory Map

Based on the Streamlit implementation, the repository follows this structural hierarchy:

```text
├── app.py                  # Main Streamlit web application script (LiSeM-Pro)
├── src/
│   └── predictor.py        # Contains the UniversalPredictor class for ML inference
├── models/                 # Directory storing the pre-trained CatBoost model files
└── requirements.txt        # Required Python dependencies (e.g., streamlit, pandas, xlsxwriter, catboost)
```

### Installation

Clone or fork the repository.

Install the required dependencies:

```Bash
pip install -r requirements.txt
```

Run the application:

```Bash
streamlit run app.py
```

### Web Application (LiSeM-Pro) Workflow

The framework features a user-friendly interface powered by Streamlit, designed to process batch predictions seamlessly.

**Model Selection:** Use the sidebar configuration menu to select the desired classification model (geotechnical_classifier or structural_classifier).

**Download Model Template:** Click the "Download Excel Template" button. The app dynamically generates and downloads an empty .xlsx file formatted with the exact feature columns required by the chosen model.

**Upload Data and Run Predictions:** Populate the downloaded template with your urban-scale screening data and upload it back into the application.

**Batch Processing & Export:** Click "Run Batch Predictions" to execute the machine-learning inference via the UniversalPredictor. Download a comprehensive results file (.xlsx) containing your original data alongside the Model_Prediction_Code, the structured Model_Prediction_Label, and the probability variables for each class (Prob_[Class]).

## 4. Credits & Attribution

**Developed and Implemented by:** Juan Carlos Pantoja (ML Modeling, Scripting, and Application Development).

**Scientific Methodology:** Alejandro Cruz Escobar (Doctoral Researcher, Civil Engineering, Universidade do Minho).

**Technical Review:** Dr. Shaghayegh Karimzadeh.

**Project Supervision:** Prof. Paulo B. Lourenço & Dr. Nicola Chieffo.

**Institution:** Universidade do Minho.

## 5. Scientific Disclaimer & License

**Intended Use:** This software is a research-oriented demonstration tool and implementation of machine-learning models intended for urban-scale screening, methodological reproduction, and academic research applications. It is not intended for real-world engineering design, decision-making, or risk assessment. It should not be used as a substitute for site-specific geotechnical investigations, structural inspections, liquefaction-triggering analyses, or detailed engineering assessment. All engineering decisions remain the responsibility of qualified professionals.

### Model Limitations & Validity

**Domain Validity:** Predictions are valid only within the ranges of earthquake magnitude, peak ground acceleration, groundwater-table depth, LPI, SLM severity, and damage conditions represented in the development databases. Predictions outside the calibration and training domains should be interpreted with caution.

**Model 3 Uncertainty:** This model was developed from a highly imbalanced reconnaissance database dominated by "no or minor" SLM observations. Predicting rare moderate and severe manifestations remains subject to greater uncertainty.

**Model 4 Limitations:** Performance estimates are affected by a limited sample size and differences in the spatial resolution of damage labels. The reported performance on the 26-observation hold-out set should be interpreted as an internal evaluation rather than broad external validation.

**Data Interdependence:** For the Chilean dataset, LPI values were estimated using field-observed SLM severity. The resulting LPI values are not statistically independent of the SLM target, which may produce optimistic performance estimates for Chilean observations in analyses involving SLM classificatio.

**Regional Transferability:** The models were trained using harmonised datasets combining observations from different earthquakes, regions, and data-collection procedures. Their direct transfer to other regions or building stocks has not been independently demonstrated. Regional compatibility should be assessed before application, and retraining may be required when local data is available.

**Liability Waiver:** This application and its contents are provided "as is", without warranty of any kind, express or implied. The developers disclaim all liability for any direct, indirect, or consequential loss or damage arising from the use of this tool. By using this application, you acknowledge that no contractual, professional, or legal relationship is established between you and the authors, affiliated institutions, or contributors.

**License:** This work is licensed under a Creative Commons Attribution-NonCommercial 3.0 Unported License.

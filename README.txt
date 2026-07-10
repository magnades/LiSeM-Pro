===========================================================================
        ML FRAMEWORK FOR RC STRUCTURE DURABILITY PREDICTION
===========================================================================

PROJECT: Life Assessment of New and Existing Reinforced Concrete Structures
INSTITUTION: TecMinho, Universidade do Minho (ICS)
PROCESS NO: ICS LE240022
DATE: March 2026

---------------------------------------------------------------------------
1. RESEARCH CONTEXT
---------------------------------------------------------------------------
This tool is part of "Task 4: Performance Evaluation of Models Based on 
Data Science." It investigates the potential of Machine Learning (ML) 
techniques to predict the evolution of key durability indicators in 
Reinforced Concrete (RC) structures.

The framework utilizes data from:
- Embedded sensors (developed by OSA) for temperature and electrical 
  resistivity (ER).
- Laboratory characterization of concrete mixes (C25/30, C40/50, and GROUT).
- Exposure conditions including chloride migration and carbonation depth.

The models aim to lay the foundation for predictive maintenance 
strategies, focusing on prediction of:
- Casting age of concrete.
- Non-Steady-State Chloride Migration Coefficients (Mnss).
- Carbonation Depths (dk).

---------------------------------------------------------------------------
2. FILE STRUCTURE & DIRECTORY MAP
---------------------------------------------------------------------------

PROJECT ROOT:
├── main.py                   : Central execution script for all predictions.
├── requirements.txt          : Python library dependencies (TensorFlow, etc).
├── README.txt                : Research context and user instructions.
├── install_dependencies.bat  : Automated setup for Windows users.
│
├── models/                   : TRAINED ARTIFACTS (Stored by Model ID)
│   │
│   ├── [TensorFlow MLP Models]
│   ├── model_2.keras ... model_7.keras       (Neural Network structures)
│   ├── preprocessor_2.pkl ... preprocessor_7.pkl (Scaling & Encoding)
│   │
│   └── [XGBoost/SKLearn Pipelines]
│       └── model_8.pkl, model_9.pkl          (Integrated Depth/Migration)
│
├── data/                     : INPUT & OUTPUT REPOSITORY
│   │
│   ├── input_data.xlsx       : Multi-sheet template. Contains one sample 
│   │                           per model derived from the research database.
│   │
│   └── [Prediction Reports]
│       └── results_2.xlsx ... results_9.xlsx : Individual output files 
│                                              containing calculated 
│                                              durability indicators.
│
└── src/                      : CORE LOGIC
    ├── predictor.py          : Universal inference engine.
    └── excel_handler.py      : Data ingestion and report generation logic.

---------------------------------------------------------------------------
3. HOW TO USE THE TOOL
---------------------------------------------------------------------------

STEP 1: ENVIRONMENT SETUP & DEPENDENCIES

    IMPORTANT: This framework is optimized for Python 3.10.x (64-bit).
    Versions 3.11 or higher are NOT compatible with TensorFlow 2.10.

    1. Open a terminal (CMD or PowerShell) in this folder.
    2. Upgrade your package installer (Critical for TensorFlow):
       python -m pip install --upgrade pip
    
    3. Install the validated requirements:
       pip install -r requirements.txt

    4. Verify the installation:
       python -c "import tensorflow as tf; print('Success: TF', tf.__version__)"

STEP 2: GENERATE THE EXCEL TEMPLATE
Run the main script for the first time:
    python main.py
The script will detect that the input file is missing and create:
    data/input_data.xlsx

STEP 3: INPUT YOUR DATA
Open 'data/input_data.xlsx'. You will see sheets named after the models 
(e.g., '2', '3', etc). 
- Fill the rows with your sensor measurements or lab results.
- IMPORTANT: Do not change the sheet names or the column headers.
- Save and CLOSE the Excel file before proceeding.

STEP 4: RUN PREDICTIONS
Run the script again:
    python main.py
The framework will process the data and generate a result file in 
the 'data/' folder (e.g., 'results_1.xlsx').

---------------------------------------------------------------------------
4. SCIENTIFIC DISCLAIMER
---------------------------------------------------------------------------
As outlined in the Task 4 report, these models are part of an 
exploratory study. The goal is to establish a machine learning 
framework that can be incrementally refined as more sensor data becomes 
available. Results should be interpreted as preliminary trends within 
the scope of the ICS LE240022 research project.

---------------------------------------------------------------------------
CREDITS & ATTRIBUTION
---------------------------------------------------------------------------
Developed and Implemented by: Juan Carlos Pantoja
(ML Modeling, Scripting, and Application Development)

Technical Review: Dr. Joaquim Tinoco
Project Supervision: Prof. Jose Antonio Campos e Matos

TecMinho, Universidade do Minho
---------------------------------------------------------------------------
===========================================================================
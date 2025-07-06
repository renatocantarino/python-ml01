# Python ML Project: Credit Risk Modeling

## Overview
This project is a machine learning pipeline for credit risk assessment, using data from a PostgreSQL database. It includes data preprocessing, feature engineering, model training (Random Forest and Neural Network), and model evaluation. The project is designed for financial data, particularly for classifying credit risk as 'bom' (good) or 'ruim' (bad).

## Features
- Data extraction from a PostgreSQL database
- Data cleaning, outlier treatment, and feature engineering
- Label encoding and feature scaling
- Feature selection using Recursive Feature Elimination (RFE)
- Model training with Random Forest and TensorFlow Neural Network
- Model evaluation and metrics reporting
- Utility scripts for data analysis and visualization

## Project Structure
```
.
├── agent.py                # Data analysis and visualization
├── model_builder.py        # Main ML pipeline: preprocessing, training, evaluation
├── model_builder.ipynb     # Jupyter notebook version of the pipeline
├── service.py              # Database connection utility
├── utils.py                # Data processing utilities
├── const.py                # SQL query and constants
├── config.yaml             # Database configuration
├── requirements.txt        # Python dependencies
├── objects/                # Model artifacts (scalers, encoders, models)
└── bankenv/                # Python virtual environment (optional)
```

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure database access**:
   - Edit `config.yaml` with your PostgreSQL credentials:
     ```yaml
     database_config:
       dbname: 'your_db'
       user: 'your_user'
       password: 'your_password'
       host: 'your_host'
     ```

## Usage
- **Run the main pipeline:**
  ```bash
  python model_builder.py
  ```
  This will:
  - Fetch data from the database
  - Preprocess and clean the data
  - Engineer features and select the most relevant ones
  - Train a Random Forest and a Neural Network
  - Save the trained models and preprocessing objects in the `objects/` directory
  - Print evaluation metrics to the console

- **Data analysis and visualization:**
  ```bash
  python agent.py
  ```
  This will generate bar plots, boxplots, and histograms for categorical and numerical variables, and print missing value statistics.

- **Jupyter Notebook:**
  You can also explore and run the pipeline step-by-step in `model_builder.ipynb`.

## Outputs
- Trained model: `objects/model.keras`
- Feature selector: `objects/selector.joblib`
- Scalers and encoders: `objects/scaler*.joblib`, `objects/labelencoder*.joblib`

## Dependencies
See `requirements.txt` for the full list. Key packages:
- pandas
- scikit-learn
- tensorflow
- joblib
- PyYAML
- psycopg2-binary
- matplotlib, seaborn (for visualization)

## Configuration
- **Database:** Set credentials in `config.yaml`.
- **SQL Query:** Defined in `const.py` as `consulta_sql`.
- **Valid professions:** Also in `const.py` as `profissoes_validas`.

## Notes
- The `objects/` directory is auto-created and stores all model artifacts.
- The `bankenv/` directory is a Python virtual environment (optional, not required if you use your own environment).
- The project is intended for educational and prototyping purposes. For production, review security and performance aspects.


## XAI

Is a process and methods thats allow humans to understand and trust de models output created by ML algs.
the more powerfull, complex and capable the model, the lower its explainability. [Neural Network]
XAI are classified as: black and white box.


-> Decision tree
    1. thats helps visualize steps, dicisions and the possible outcomes od each decision

-> Linear Regression
    1. statistical technique used to find the relationship between variables and possible outcomes


uses:

1. medical diagnosis
2. credit risk assessment
3. laws decisions
4. autonomous vehicles
5. HR




## License
MIT License (add your license here if different) 
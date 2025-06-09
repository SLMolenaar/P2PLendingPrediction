# P2P Lending Risk Prediction

A machine learning pipeline and web application for predicting loan default risk in peer-to-peer lending platforms. This project includes both a data processing/modeling pipeline and an interactive web interface for risk assessment.

## Features

- **Interactive Web Interface**: User-friendly dashboard for risk assessment
- **Real-time Predictions**: Get instant risk scores for loan applications
- **Model Insights**: View feature importance and model performance metrics
- **Data Processing**: Handles missing values, feature engineering, and data leakage prevention
- **Fraud Detection**: Identifies potential fraudulent applications using anomaly detection
- **Modeling**: Implements multiple ML models (Logistic Regression, Random Forest, XGBoost)
- **Risk Scoring**: Generates risk scores (0-100) for loan applications

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SLMolenaar/P2PLendingPrediction.git
   cd P2PLendingPrediction
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web Application

1. Ensure your virtual environment is activated
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. The application will open automatically in your default web browser at `http://localhost:8501`

## Using the Web Application

1. **Predict Risk Page**:
   - Fill in the loan application details
   - Click "Assess Risk" to get a risk score
   - View the risk level and recommendations

2. **Model Insights Page**:
   - View model performance metrics
   - See feature importance
   - Compare different model performances

## Data Requirements

Place your loan data in the `data/` directory. The expected format is a CSV file containing loan application data with features like:
- `loan_amnt`: Loan amount
- `annual_inc`: Annual income
- `emp_length`: Employment length
- `purpose`: Loan purpose
- `loan_status`: Target variable (e.g., 'Fully Paid', 'Charged Off')

### Required Features:
- `loan_amnt`: Requested loan amount ($)
- `annual_inc`: Annual income ($)
- `emp_length`: Employment length (years)
- `int_rate`: Interest rate of the loan
- `dti`: Debt-to-income ratio
- `revol_util`: Revolving line utilization rate
- `purpose`: Loan purpose category
- `loan_status`: Target variable (e.g., 'Fully Paid', 'Charged Off')

### Example Data Format:
```csv
loan_amnt,annual_inc,emp_length,int_rate,dti,revol_util,purpose,loan_status
10000,75000,5,12.5,18.5,45,debt_consolidation,Fully Paid
5000,45000,2,15.2,22.1,67,credit_card,Charged Off
```

## Usage

Run the main pipeline:
```bash
python -m src.Main
```

The pipeline will:
1. Load and preprocess the data
2. Perform feature engineering
3. Train and evaluate multiple models
4. Generate risk scores
5. Output performance metrics and insights

## Project Structure

```
P2PLendingPrediction/
├── data/                   # Data directory (not version controlled)
├── src/                    # Source code
│   ├── __init__.py         # Makes src a Python package
│   ├── Main.py             # Main pipeline
│   ├── DataLoader.py       # Data loading and preprocessing
│   ├── Preprocessor.py     # Feature engineering and data cleaning
│   └── Model.py            # Model training and evaluation
├── .gitignore             # Specifies intentionally untracked files
├── setup.py               # Package configuration
└── README.md              # This file
```

## Model Performance

Expected performance metrics:
- AUC-ROC: 0.65-0.85 (realistic for credit risk models)
- Key risk factors identified:
  - Debt-to-income ratio
  - Credit utilization
  - Employment stability
  - Loan purpose

## Acknowledgments

- Lending Club for the loan data

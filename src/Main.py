import os
import warnings
from typing import Dict, List, Tuple, Optional

# Import local modules
from src.DataLoader import DataLoader
from src.Preprocessor import Preprocessor
from src.Model import Model

# Suppress warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # Configuration
    DATA_PATH = '../data/accepted_2007_to_2018Q4_extracted.csv'

    try:
        print("=== STARTING RISK SCORING PIPELINE ===")

        # Step 1: Load and explore data
        print("\n1. Loading and exploring data...")
        data_loader = DataLoader(DATA_PATH)
        df = data_loader.load_data()
        data_loader.explore_data()
        df = data_loader.remove_data_leakage()

        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        preprocessor = Preprocessor()
        df_engineered = preprocessor.engineer_features(df)
        df_with_fraud, fraud_indicators = preprocessor.detect_fraud_patterns(df_engineered)
        X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(df_with_fraud)

        # Step 3: Train and evaluate models
        print("\n3. Training and evaluating models...")
        model = Model()
        model.set_data(X_train, X_test, y_train, y_test)
        trained_models = model.train_models()
        results = model.evaluate_models(preprocessor.feature_names)

        # Step 4: Create risk scoring system
        print("\n4. Creating risk scoring system...")
        risk_results = model.create_risk_score(method='ensemble')

        # Step 5: Generate insights
        print("\n5. Generating insights...")
        model.generate_insights()

        print("\n[SUCCESS] Data leakage-free risk scoring pipeline completed successfully!")
        print("Expected AUC scores: 0.65-0.85 (realistic for credit risk models)")

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
#!/usr/bin/env python3
"""
Script to reset and retrain models with proper feature names.
Run this script to fix the feature names mismatch issue.

Usage:
    python reset_and_retrain.py
"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import your modules
try:
    from src.DataLoader import DataLoader
    from src.Preprocessor import Preprocessor
    from src.Model import Model

    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)


def reset_models():
    """Reset saved models and preprocessed data"""
    saved_models_dir = current_dir / 'saved_models'

    if saved_models_dir.exists():
        print(f"🗑️  Removing existing saved models directory: {saved_models_dir}")
        shutil.rmtree(saved_models_dir)

    saved_models_dir.mkdir(exist_ok=True)
    print(f"📁 Created fresh saved models directory: {saved_models_dir}")


def retrain_models():
    """Retrain models with proper feature names"""
    print("\n🔄 Starting model retraining with proper feature names...")

    # Configuration
    data_path = current_dir / 'data' / 'accepted_2007_to_2018Q4_extracted.csv'

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists in the correct location.")
        return False

    try:
        # Step 1: Load and preprocess data
        print("\n1️⃣ Loading and preprocessing data...")
        data_loader = DataLoader(str(data_path))
        df = data_loader.load_data()
        df = data_loader.remove_data_leakage()

        # Step 2: Feature engineering and preprocessing
        print("\n2️⃣ Engineering features...")
        preprocessor = Preprocessor()
        df_engineered = preprocessor.engineer_features(df)
        df_with_fraud, _ = preprocessor.detect_fraud_patterns(df_engineered)

        # Step 3: Prepare data for modeling (this will set feature_names in preprocessor)
        print("\n3️⃣ Preparing data for modeling...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(df_with_fraud)

        # Step 4: Train models with proper feature names
        print("\n4️⃣ Training models with proper feature names...")
        model = Model()

        # Set data with feature names from preprocessor
        feature_names = getattr(preprocessor, 'feature_names', None)
        if feature_names:
            print(f"✅ Using {len(feature_names)} feature names from preprocessor")
            print(f"📋 First 5 features: {feature_names[:5]}")
        else:
            print("⚠️  No feature names found in preprocessor, using generic names")
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        model.set_data(X_train, X_test, y_train, y_test, feature_names)

        # Train the models
        trained_models = model.train_models()

        # Save the model
        print("\n6️⃣ Saving trained model...")
        model_path = current_dir / 'saved_models' / 'trained_model.joblib'
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'created_at': pd.Timestamp.now().isoformat()
        }

        import joblib
        joblib.dump(model_data, model_path)
        print(f"✅ Model saved to: {model_path}")

        # Evaluate models
        print("\n5️⃣ Evaluating models...")
        results = model.evaluate_models()

        # Test prediction to verify everything works
        print("\n7️⃣ Testing prediction functionality...")
        test_data = pd.DataFrame({
            'loan_amnt': [10000],
            'term': [36],
            'int_rate': [10.0],
            'annual_inc': [60000],
            'dti': [15.0],
            'fico_range_high': [720],
            'fico_range_low': [700]
        })

        try:
            risk_scores = model.predict_risk(test_data)
            print(f"✅ Test prediction successful! Risk score: {risk_scores[0]}")
            return True
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Error during model retraining: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to reset and retrain models"""
    print("🚀 P2P Lending Model Reset and Retrain Script")
    print("=" * 50)

    # Step 1: Reset existing models
    print("\n🧹 Step 1: Resetting existing models...")
    reset_models()

    # Step 2: Retrain models
    print("\n🏋️ Step 2: Retraining models with proper feature names...")
    success = retrain_models()

    if success:
        print("\n🎉 SUCCESS! Models have been successfully retrained with proper feature names.")
        print("\n📝 Next steps:")
        print("1. Run your Streamlit app: streamlit run app.py")
        print("2. Initialize the system using the button in the app")
        print("3. Try making a prediction to verify everything works")
        print("\n💡 The feature names mismatch error should now be resolved!")
    else:
        print("\n💥 FAILED! There was an error during the retraining process.")
        print("Please check the error messages above and try again.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
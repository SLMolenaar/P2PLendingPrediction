import streamlit as st
import pandas as pd
import joblib
import os
import sys
import logging
from pathlib import Path
from functools import lru_cache
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory (where app.py is located)
CURRENT_DIR = Path(__file__).parent
# Add src directory to path
sys.path.append(str(CURRENT_DIR / 'src'))

# Import your existing modules with error handling
try:
    from DataLoader import DataLoader
    from Preprocessor import Preprocessor
    from Model import Model
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Constants - using your actual directory structure
MODEL_DIR = CURRENT_DIR / 'saved_models'
MODEL_DIR.mkdir(exist_ok=True)

# Set page config
st.set_page_config(
    page_title="P2P Lending Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    """Load modern custom CSS"""
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f9f9fb;
        }

        .stButton>button {
            background: linear-gradient(135deg, #4CAF50, #43a047);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: background 0.3s, transform 0.2s;
            cursor: pointer;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, #45a049, #388e3c);
            transform: scale(1.02);
        }

        .risk-card {
            padding: 1.25rem;
            border-radius: 12px;
            margin: 1.2rem 0;
            text-align: center;
            font-size: 1rem;
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .very-low-risk { background-color: #e6f4ea; color: #1b5e20; }
        .low-risk { background-color: #e0f7fa; color: #006064; }
        .medium-risk { background-color: #fff9e6; color: #795548; }
        .high-risk { background-color: #fdecea; color: #c62828; }
        .very-high-risk { background-color: #f8d7da; color: #b71c1c; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(ttl=3600, show_spinner="Loading and preprocessing data...")
def load_data():
    """Load and preprocess data with caching and proper error handling"""
    try:
        data_path = CURRENT_DIR / 'data' / 'accepted_2007_to_2018Q4_extracted.csv'
        preprocessed_path = MODEL_DIR / 'preprocessed_data.joblib'

        # Check if data file exists
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return None, None, None, None, None

        # Try to load preprocessed data first
        if preprocessed_path.exists():
            try:
                data = joblib.load(preprocessed_path)
                logger.info("Loaded preprocessed data from cache")
                return (
                    data['X_train'],
                    data['X_test'],
                    data['y_train'],
                    data['y_test'],
                    data.get('feature_names', [])
                )
            except Exception as e:
                logger.warning(f"Could not load preprocessed data: {e}")

        # Process from scratch
        logger.info("Processing data from scratch...")
        data_loader = DataLoader(str(data_path))
        df = data_loader.load_data()

        if df is None or df.empty:
            raise ValueError("Failed to load data or data is empty")

        df = data_loader.remove_data_leakage()

        preprocessor = Preprocessor()
        df_engineered = preprocessor.engineer_features(df)
        df_with_fraud, _ = preprocessor.detect_fraud_patterns(df_engineered)

        # Split data
        X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(df_with_fraud)

        # Get feature names from preprocessor
        feature_names = getattr(preprocessor, 'feature_names', [])

        # Save preprocessed data for faster loading
        joblib.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }, preprocessed_path)

        logger.info("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test, feature_names

    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        return None, None, None, None, None


class P2PLendingApp:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_or_train_models(self, X_train, X_test, y_train, y_test, feature_names=None):
        """Load trained models or train new ones if not found"""
        # Use the correct path for your saved model
        model_path = MODEL_DIR / 'trained_model.joblib'

        logger.info(f"Looking for model at: {model_path}")

        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                model = model_data['model']

                # Restore feature names
                if 'feature_names' in model_data:
                    restored_feature_names = model_data['feature_names']
                    model.feature_names = restored_feature_names
                    logger.info(f"Restored {len(restored_feature_names)} feature names from saved model")
                elif feature_names is not None:
                    model.feature_names = feature_names
                    logger.info(f"Using provided feature names: {len(feature_names)} features")

                # Set data with proper feature names
                model.set_data(X_train, X_test, y_train, y_test, model.feature_names)
                logger.info("Loaded trained model from cache")
                return model
            except Exception as e:
                logger.warning(f"Error loading saved model: {str(e)}. Training new model...")

        # Train new model
        logger.info("Training new model...")
        model = Model()

        # Set data with feature names BEFORE training
        model.set_data(X_train, X_test, y_train, y_test, feature_names)

        # Train the models
        model.train_models()

        # Save the trained model with feature names
        try:
            model_to_save = {
                'model': model,
                'feature_names': feature_names,
                'created_at': datetime.now().isoformat()
            }
            joblib.dump(model_to_save, model_path)
            logger.info(f"Model training completed and saved to: {model_path}")
            logger.info(f"Saved {len(feature_names) if feature_names else 0} feature names")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")

        return model

    def initialize_models(self):
        """Initialize models and data with comprehensive error handling"""
        try:
            with st.spinner('Loading data (this may take a few minutes on first run)...'):
                X_train, X_test, y_train, y_test, feature_names = load_data()

                if X_train is None:
                    st.error("❌ Failed to load data. Please check:")
                    st.markdown("""
                    - Ensure `data/accepted_2007_to_2018Q4_extracted.csv` exists
                    - Check file permissions
                    - Verify file format and content
                    """)
                    return False

                with st.spinner('Initializing models...'):
                    self.model = self.load_or_train_models(X_train, X_test, y_train, y_test, feature_names)
                    if self.model is None:
                        st.error("❌ Failed to initialize models.")
                        return False

                # Store data in instance variables
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                self.feature_names = feature_names

                return True

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            st.error(f"❌ Error initializing models: {str(e)}")
            return False


def get_risk_level_info(risk_score):
    """Get risk level information and styling"""
    if risk_score < 300:
        return "Very Low", "very-low-risk", "✅", "This loan application appears to be very low risk."
    elif risk_score < 500:
        return "Low", "low-risk", "✅", "This loan application appears to be low risk."
    elif risk_score < 700:
        return "Medium", "medium-risk", "⚠️", "This loan application has moderate risk."
    elif risk_score < 800:
        return "High", "high-risk", "❌", "This loan application is considered high risk."
    else:
        return "Very High", "very-high-risk", "❌❌", "This loan application is considered very high risk."


def validate_inputs(loan_amnt, term, int_rate, annual_inc, dti, fico_score):
    """Validate user inputs"""
    errors = []

    if loan_amnt < 1000 or loan_amnt > 40000:
        errors.append("Loan amount must be between $1,000 and $40,000")
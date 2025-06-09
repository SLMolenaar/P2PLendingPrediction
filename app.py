import streamlit as st
import pandas as pd
import joblib
import os
import sys
import logging
from pathlib import Path
from functools import lru_cache
import numpy as np

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

        # Save preprocessed data for faster loading
        joblib.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': getattr(preprocessor, 'feature_names', [])
        }, preprocessed_path)

        logger.info("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test, getattr(preprocessor, 'feature_names', [])

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
                    model.feature_names = model_data['feature_names']
                elif feature_names is not None:
                    model.feature_names = feature_names

                model.set_data(X_train, X_test, y_train, y_test)
                logger.info("Loaded trained model from cache")
                return model
            except Exception as e:
                logger.warning(f"Error loading saved model: {str(e)}. Training new model...")

        # Train new model
        logger.info("Training new model...")
        model = Model()
        model.set_data(X_train, X_test, y_train, y_test)

        # Train the models
        model.train_models()

        if feature_names is not None:
            model.feature_names = feature_names

        # Save the trained model
        try:
            joblib.dump({'model': model, 'feature_names': feature_names}, model_path)
            logger.info(f"Model training completed and saved to: {model_path}")
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

    if int_rate < 5.0 or int_rate > 30.0:
        errors.append("Interest rate must be between 5% and 30%")

    if annual_inc < 1000:
        errors.append("Annual income must be at least $1,000")

    if dti < 0 or dti > 50:
        errors.append("Debt-to-income ratio must be between 0 and 50")

    if fico_score < 300 or fico_score > 850:
        errors.append("FICO score must be between 300 and 850")

    return errors


def show_prediction_interface(app):
    """Show the prediction interface with improved UX"""
    st.header("🔍 Loan Application Risk Assessment")

    # Create a form for user input
    with st.form("loan_form"):
        st.subheader("Enter Loan Application Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**💰 Loan Information**")
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=40000,
                value=10000,
                help="Amount requested for the loan"
            )
            term = st.selectbox("Loan Term", ["36 months", "60 months"])
            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=5.0,
                max_value=30.0,
                value=10.0,
                step=0.1,
                help="Annual interest rate for the loan"
            )

        with col2:
            st.markdown("**👤 Borrower Information**")
            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=1000,
                value=60000,
                step=1000,
                help="Borrower's annual income"
            )
            dti = st.number_input(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=0.1,
                help="Monthly debt payments / Monthly income"
            )
            fico_score = st.number_input(
                "FICO Score",
                min_value=300,
                max_value=850,
                value=700,
                help="Credit score (higher is better)"
            )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("🎯 Assess Risk", use_container_width=True)

    if submitted:
        # Validate inputs
        errors = validate_inputs(loan_amnt, term, int_rate, annual_inc, dti, fico_score)

        if errors:
            st.error("❌ Please fix the following errors:")
            for error in errors:
                st.write(f"• {error}")
            return

        # Create input DataFrame
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': [term],
            'int_rate': [int_rate],
            'annual_inc': [annual_inc],
            'dti': [dti],
            'fico_range_high': [fico_score],
            'fico_range_low': [fico_score - 20],
            'revol_util': [min(dti * 2, 100)],
            'open_acc': [5],
            'total_acc': [10],
            'pub_rec': [0],
            'delinq_2yrs': [0],
            'inq_last_6mths': [1]
        })

        try:
            with st.spinner("Analyzing risk..."):
                risk_scores = app.model.predict_risk(input_data)
                risk_score = int(risk_scores[0])

            # Display results
            st.subheader("📊 Risk Assessment Results")

            # Get risk level info
            risk_level, risk_class, icon, description = get_risk_level_info(risk_score)

            # Create risk card
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h2>{icon} {risk_level} Risk</h2>
                <h1>{risk_score}/1000</h1>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

            # Risk gauge visualization
            gauge_value = risk_score / 10  # Convert to 0-100 scale
            st.metric("Risk Score", f"{risk_score}/1000", delta=f"{gauge_value:.1f}% risk level")

            # Recommendations
            with st.expander("💡 Detailed Recommendations", expanded=True):
                if risk_score < 200:
                    st.success("**Recommendation:** ✅ APPROVE")
                    st.write("This application meets all standard criteria with minimal risk.")
                elif risk_score < 400:
                    st.info("**Recommendation:** ✅ APPROVE with standard terms")
                    st.write("This application is within acceptable risk parameters.")
                elif risk_score < 600:
                    st.warning("**Recommendation:** ⚠️ CONDITIONAL APPROVAL")
                    st.write("Consider additional verification or adjusted terms:")
                    st.write("• Request additional documentation")
                    st.write("• Consider higher interest rate")
                    st.write("• Require co-signer if possible")
                elif risk_score < 800:
                    st.error("**Recommendation:** ❌ DECLINE or require significant risk mitigation")
                    st.write("High risk factors present. If considering approval:")
                    st.write("• Require substantial down payment")
                    st.write("• Implement strict monitoring")
                    st.write("• Consider collateral requirements")
                else:
                    st.error("**Recommendation:** ❌ DECLINE")
                    st.write("Risk level is too high for standard approval.")

                # Risk factors analysis
                st.subheader("🔍 Risk Factor Analysis")
                factors = []

                if dti > 30:
                    factors.append(f"• High debt-to-income ratio ({dti:.1f}%)")
                if int_rate > 15:
                    factors.append(f"• High interest rate ({int_rate:.1f}%)")
                if fico_score < 650:
                    factors.append(f"• Low credit score ({fico_score})")
                if loan_amnt > annual_inc * 0.5:
                    factors.append(f"• High loan-to-income ratio")
                if term == "60 months":
                    factors.append("• Extended loan term increases risk")

                if factors:
                    st.write("**Risk factors identified:**")
                    for factor in factors:
                        st.write(factor)
                else:
                    st.write("✅ No major risk factors identified")

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            st.error(f"❌ Error making prediction: {str(e)}")
            st.write("Please check your inputs and try again.")


def show_model_insights(app):
    """Show model performance and insights with better error handling"""
    st.header("📈 Model Performance & Insights")

    if not hasattr(app, 'model') or app.model is None:
        st.warning("⚠️ Model not initialized. Please go to the Predict Risk page first.")
        return

    try:
        # Show model metrics
        st.subheader("🎯 Model Performance")

        if hasattr(app.model, 'models') and app.model.models:
            # Display basic info
            st.info(f"📊 Models trained: {len(app.model.models)}")
            st.info(f"🔢 Features used: {len(app.feature_names) if app.feature_names else 'Unknown'}")

            # Show feature importance if available
            if hasattr(app.model, 'feature_importance') and app.model.feature_importance:
                st.subheader("🔍 Feature Importance")

                # Get Random Forest feature importance
                if 'random_forest' in app.model.feature_importance:
                    importance_df = app.model.feature_importance['random_forest'].head(15)
                    st.bar_chart(importance_df.set_index('feature')['importance'])

                    with st.expander("View Top Features"):
                        st.dataframe(importance_df)

            # Model comparison
            st.subheader("🤖 Model Ensemble")
            st.write("This system uses an ensemble of three models:")

            models_info = pd.DataFrame({
                'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
                'Weight': ['30%', '30%', '40%'],
                'Purpose': [
                    'Interpretable baseline model',
                    'Feature importance analysis',
                    'High-performance predictions'
                ]
            })
            st.dataframe(models_info, use_container_width=True)

        else:
            st.warning("⚠️ Model performance data not available.")

    except Exception as e:
        logger.error(f"Error in model insights: {str(e)}")
        st.error(f"❌ Error loading model insights: {str(e)}")


def main():
    # Load CSS
    load_css()

    # Header
    st.title("💰 P2P Lending Risk Predictor")
    st.markdown("### Intelligent risk assessment for peer-to-peer lending applications")

    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = P2PLendingApp()

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    # Show initialization status
    if not st.session_state.initialized:
        with st.container():
            st.info("🚀 Initializing application...")
            st.warning("⚠️ First-time loading may take a few minutes as it processes data and trains models.")

            if st.button("🔄 Initialize System", type="primary"):
                if st.session_state.app.initialize_models():
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize system. Please check the logs and try again.")
        return

    # Main navigation
    st.sidebar.header("📍 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🎯 Predict Risk", "📊 Model Insights"],
        help="Choose what you'd like to do"
    )

    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
    st.sidebar.success("✅ Models loaded")
    if hasattr(st.session_state.app, 'feature_names') and st.session_state.app.feature_names:
        st.sidebar.info(f"📊 {len(st.session_state.app.feature_names)} features")

    # Main content
    if page == "🎯 Predict Risk":
        show_prediction_interface(st.session_state.app)
    else:
        show_model_insights(st.session_state.app)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This application uses machine learning to assess 
    the risk of P2P lending applications based on 
    borrower information and loan characteristics.
    """)


if __name__ == "__main__":
    main()
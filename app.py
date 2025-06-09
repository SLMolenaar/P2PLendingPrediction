import streamlit as st
import pandas as pd
import joblib
import os
import sys
import time
from pathlib import Path
from functools import lru_cache

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import your existing modules
from DataLoader import DataLoader
from Preprocessor import Preprocessor
from Model import Model

# Constants
MODEL_DIR = Path('saved_models')
MODEL_DIR.mkdir(exist_ok=True)

# Set page config
st.set_page_config(
    page_title="P2P Lending Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS"""
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stSelectbox, .stNumberInput, .stTextInput {
            margin-bottom: 1rem;
        }
        .success-msg {
            color: #4CAF50;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(ttl=3600, show_spinner="Loading and preprocessing data...")
def load_data():
    """Load and preprocess data with caching"""
    try:
        data_path = os.path.join('data', 'accepted_2007_to_2018Q4_extracted.csv')
        preprocessed_path = MODEL_DIR / 'preprocessed_data.joblib'
        
        # Try to load preprocessed data first
        if preprocessed_path.exists():
            try:
                data = joblib.load(preprocessed_path)
                return (
                    data['X_train'],
                    data['X_test'],
                    data['y_train'],
                    data['y_test'],
                    data.get('feature_names', [])
                )
            except Exception as e:
                print(f"Warning: Could not load preprocessed data: {e}")
        
        # If no preprocessed data exists or error loading, process from scratch
        data_loader = DataLoader(data_path)
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
        
        return X_train, X_test, y_train, y_test, getattr(preprocessor, 'feature_names', [])
        
    except Exception as e:
        st.error(f"Error in load_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

class P2PLendingApp:
    def __init__(self):
        self.data_loader = None
        self.preprocessor = None
        self.model = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_or_train_models(self, X_train, X_test, y_train, y_test, feature_names=None):
        """Load trained models or train new ones if not found"""
        model_path = MODEL_DIR / 'trained_model.joblib'
        
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                model = model_data['model']
                # Restore feature_names if present in file, otherwise use argument
                if hasattr(model, 'feature_names') and getattr(model, 'feature_names', None):
                    pass
                elif feature_names is not None:
                    model.feature_names = feature_names
                else:
                    # Try to load from model_data if available
                    model.feature_names = model_data.get('feature_names', None)
                model.set_data(X_train, X_test, y_train, y_test)
                return model
            except Exception as e:
                st.warning(f"Error loading saved model: {str(e)}. Training new model...")
        
        # Train new model if not found or error loading
        model = Model()
        model.set_data(X_train, X_test, y_train, y_test)
        model.train_models()
        if feature_names is not None:
            model.feature_names = feature_names
        # Save the trained model and feature_names
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        joblib.dump({'model': model, 'feature_names': feature_names}, model_path)
        return model
    
    def initialize_models(self):
        """Initialize models and data"""
        try:
            with st.spinner('Loading data (this may take a few minutes on first run)...'):
                X_train, X_test, y_train, y_test, feature_names = load_data()
                
                if X_train is None:
                    st.error("Failed to load data. Please check the data file and try again.")
                    return False
                
                with st.spinner('Initializing models...'):
                    self.model = self.load_or_train_models(X_train, X_test, y_train, y_test, feature_names=feature_names)
                    if self.model is None:
                        st.error("Failed to initialize models.")
                        return False
                # Store data in instance variables
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                # Store feature names if available
                if feature_names is not None:
                    self.model.feature_names = feature_names
                # Set model data if the model has the method
                if hasattr(self.model, 'set_data'):
                    self.model.set_data(X_train, X_test, y_train, y_test)
                return True
                
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            return False

def main():
    # Load CSS
    load_css()
    
    st.title("💰 P2P Lending Risk Predictor")
    st.markdown("### Predict the risk level of P2P loan applications")
    
    # Add a warning about first-time loading
    st.warning("⚠️ First-time loading may take a few minutes as it processes the data and trains the models. Subsequent loads will be much faster.")
    
    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = P2PLendingApp()
    
    # Show loading state
    if 'initialized' not in st.session_state:
        if st.session_state.app.initialize_models():
            st.session_state.initialized = True
            st.rerun()  # Rerun to clear the loading state
        else:
            st.error('Failed to initialize models. Please check the console for error details.')
            st.stop()
    
    # Clear any previous error messages if we got here
    st.session_state.error = None
    
    # Main app interface
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Predict Risk", "Model Insights"])
    
    if page == "Predict Risk":
        show_prediction_interface(st.session_state.app)
    else:
        show_model_insights(st.session_state.app)

def show_prediction_interface(app):
    """Show the prediction interface"""
    st.header("Loan Application Risk Assessment")
    
    # Create a form for user input
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=10000)
            term = st.selectbox("Loan Term", ["36 months", "60 months"])
            int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=10.0, step=0.1)
            
        with col2:
            annual_inc = st.number_input("Annual Income ($)", min_value=20000, value=60000, step=1000)
            dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
            fico_range_high = st.number_input("FICO Score", min_value=300, max_value=850, value=700)
        
        submitted = st.form_submit_button("Assess Risk")
    
    if submitted:
        # Create input DataFrame with the features we collect from the user
        # The model's predict_risk method will handle adding any missing features with default values
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': term,  # Keep as string (e.g., '36 months')
            'int_rate': [int_rate],
            'annual_inc': [annual_inc],
            'dti': [dti],
            'fico_range_high': [fico_range_high],
            'fico_range_low': [fico_range_high - 20],  # Typically FICO ranges are about 20 points apart
            'revol_util': [dti * 2],  # Rough estimate
            'open_acc': [5],  # Reasonable default
            'total_acc': [10],  # Reasonable default
            'pub_rec': [0],  # No public records by default
            'delinq_2yrs': [0],  # No delinquencies by default
            'inq_last_6mths': [1]  # Reasonable default
        })
        
        # Get prediction
        try:
            risk_scores = app.model.predict_risk(input_data)
            risk_score = risk_scores[0]  # Get the first (and only) prediction
            
            # Display results
            st.subheader("Risk Assessment Results")
            
            # Visualize risk score (0-1000 scale)
            st.metric("Risk Score", f"{risk_score}/1000")
            
            # Risk level interpretation (0-1000 scale)
            if risk_score < 200:
                st.success("✅ Very Low Risk: This loan application appears to be very low risk.")
                risk_color = "green"
                recommendation = "This application is well within acceptable risk parameters."
            elif risk_score < 400:
                st.success("✅ Low Risk: This loan application appears to be low risk.")
                risk_color = "lightgreen"
                recommendation = "This application meets standard risk criteria."
            elif risk_score < 600:
                st.warning("⚠️ Medium Risk: This loan application has moderate risk.")
                risk_color = "orange"
                recommendation = "Consider additional review before approval."
            elif risk_score < 800:
                st.error("❌ High Risk: This loan application is considered high risk.")
                risk_color = "red"
                recommendation = "Proceed with caution. Additional review required."
            else:
                st.error("❌❌ Very High Risk: This loan application is considered very high risk.")
                risk_color = "darkred"
                recommendation = "Strongly reconsider approval. Significant risk factors present."
            
            # Additional recommendations
            with st.expander("Recommendations"):
                st.write(recommendation)
                if risk_score >= 400:
                    st.write("Consider the following to reduce risk:")
                    st.write("- Increase the down payment")
                    st.write("- Improve credit score")
                    st.write("- Reduce existing debt")
                    st.write("- Provide additional collateral")
                    st.write("- Consider a co-signer")
                    
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def show_model_insights(app):
    """Show model performance and insights"""
    st.header("Model Performance & Insights")
    
    if not hasattr(app, 'model') or app.model is None:
        st.warning("Model not initialized. Please go to the Predict Risk page first.")
        return
    
    # Show model metrics
    st.subheader("Model Performance")
    
    # Get evaluation results (you'll need to implement this in your Model class)
    try:
        results = app.model.evaluate_models(app.preprocessor.feature_names)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AUC Score", f"{results.get('auc', 0):.3f}")
        with col2:
            st.metric("Accuracy", f"{results.get('accuracy', 0):.1%}")
        with col3:
            st.metric("Precision", f"{results.get('precision', 0):.1%}")
        
        # Feature importance
        st.subheader("Feature Importance")
        st.write("The following features are most important in determining loan risk:")
        
        # Get feature importance (you'll need to implement this in your Model class)
        feature_importance = app.model.get_feature_importance()
        if feature_importance is not None:
            st.bar_chart(feature_importance.head(10))
        
    except Exception as e:
        st.error(f"Error loading model insights: {str(e)}")
    
    # Add model comparison if available
    if hasattr(app.model, 'model_comparison'):
        st.subheader("Model Comparison")
        st.dataframe(app.model.model_comparison)

if __name__ == "__main__":
    main()

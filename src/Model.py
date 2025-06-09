import os
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb


# Suppress warnings
warnings.filterwarnings('ignore')


class Model:
    """Handles model training, evaluation, and risk scoring"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def set_data(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """Set training and test data"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

    def train_models(self) -> Dict:
        """Train multiple models for ensemble approach"""
        print("\n[INFO] Training models...")

        if self.X_train is None:
            raise ValueError("Data must be set first. Call set_data()")

        # Logistic Regression (interpretable baseline)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['logistic_regression'] = lr

        # Random Forest (feature importance)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf

        # XGBoost (high performance)
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model

        print(f"[INFO] Trained {len(self.models)} models")
        return self.models

    def evaluate_models(self, feature_names: List[str] = None) -> Dict:
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION (NO DATA LEAKAGE) ===")

        if not self.models:
            raise ValueError("Models must be trained first. Call train_models()")

        results = {}

        for name, model in self.models.items():
            print(f"\n--- {name.upper()} ---")

            # Use scaled data for logistic regression, original for tree-based
            X_test_eval = self.X_test_scaled if name == 'logistic_regression' else self.X_test

            # Predictions
            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]

            # Metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)

            print(f"AUC Score: {auc_score:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))

            # Store results
            results[name] = {
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_') and feature_names:
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                self.feature_importance[name] = importance
                print(f"\nTop 10 Features:")
                print(importance.head(10))

        return results

    def create_risk_score(self, method: str = 'ensemble') -> pd.DataFrame:
        """Create final risk scoring system"""
        print(f"\n[INFO] Creating risk scoring system using {method} method...")

        if not self.models:
            raise ValueError("Models must be trained first. Call train_models()")

        if method == 'ensemble':
            # Weighted ensemble of all models
            weights = {'logistic_regression': 0.3, 'random_forest': 0.3, 'xgboost': 0.4}

            ensemble_proba = np.zeros(len(self.y_test))
            for name, model in self.models.items():
                X_test_eval = self.X_test_scaled if name == 'logistic_regression' else self.X_test
                proba = model.predict_proba(X_test_eval)[:, 1]
                ensemble_proba += weights[name] * proba

            risk_scores = (ensemble_proba * 1000).astype(int)

        else:
            # Use single best model
            best_auc = 0
            best_model_name = None
            for name, model in self.models.items():
                X_test_eval = self.X_test_scaled if name == 'logistic_regression' else self.X_test
                proba = model.predict_proba(X_test_eval)[:, 1]
                auc = roc_auc_score(self.y_test, proba)
                if auc > best_auc:
                    best_auc = auc
                    best_model_name = name

            best_model = self.models[best_model_name]
            X_test_eval = self.X_test_scaled if best_model_name == 'logistic_regression' else self.X_test
            proba = best_model.predict_proba(X_test_eval)[:, 1]
            risk_scores = (proba * 1000).astype(int)
            ensemble_proba = proba

        # Create risk categories
        risk_categories = pd.cut(risk_scores,
                                 bins=[0, 200, 400, 600, 800, 1000],
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Results summary
        results_df = pd.DataFrame({
            'actual_default': self.y_test.values,
            'risk_score': risk_scores,
            'risk_category': risk_categories,
            'ensemble_probability': ensemble_proba
        })

        print("\nRisk Score Distribution:")
        print(results_df['risk_category'].value_counts())

        print("\nDefault Rate by Risk Category:")
        default_by_risk = results_df.groupby('risk_category')['actual_default'].mean()
        print(default_by_risk)

        return results_df

    def predict_risk(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Predict risk score for new loan applications
        
        Args:
            input_data: DataFrame containing loan application features
            
        Returns:
            np.ndarray: Array of risk scores (0-1000)
        """
        if not self.models:
            raise ValueError("Models must be trained first. Call train_models()")
            
        if not hasattr(self, 'feature_names'):
            raise ValueError("Feature names not found. The model may not have been properly trained.")
            
        # Create a DataFrame with all required features and default values
        X_pred = pd.DataFrame(index=input_data.index)
        
        # Add all required features with default values
        for feature in self.feature_names:
            if feature in input_data.columns:
                X_pred[feature] = input_data[feature]
            else:
                # Set default values based on feature type/meaning
                if feature in ['loan_amnt', 'funded_amnt', 'funded_amnt_inv']:
                    X_pred[feature] = input_data.get('loan_amnt', 10000)
                elif feature == 'int_rate':
                    X_pred[feature] = input_data.get('int_rate', 10.0)
                elif feature == 'term':
                    term = input_data.get('term', '36 months')
                    X_pred[feature] = int(term.split()[0]) if isinstance(term, str) else term
                elif feature == 'annual_inc':
                    X_pred[feature] = input_data.get('annual_inc', 60000)
                elif feature == 'dti':
                    X_pred[feature] = input_data.get('dti', 15.0)
                elif feature in ['fico_range_low', 'fico_range_high']:
                    fico = input_data.get('fico_range_high', input_data.get('fico_range_low', 700))
                    X_pred[feature] = fico
                elif feature in ['revol_util', 'bc_util', 'il_util', 'all_util']:
                    X_pred[feature] = input_data.get(feature, 30.0)  # Default utilization percentage
                elif feature in ['open_acc', 'total_acc', 'pub_rec', 'delinq_2yrs', 'inq_last_6mths']:
                    X_pred[feature] = input_data.get(feature, 0)
                else:
                    X_pred[feature] = 0  # Default for other numeric features
        
        # Ensure all features are numeric
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Ensure we only have the features the model expects
        X_pred = X_pred[self.feature_names]
            
        # Get ensemble predictions
        ensemble_proba = np.zeros(len(X_pred))
        weights = {'logistic_regression': 0.3, 'random_forest': 0.3, 'xgboost': 0.4}
        
        for name, model in self.models.items():
            # Scale features for logistic regression
            if name == 'logistic_regression':
                X_input = self.scaler.transform(X_pred)
            else:
                X_input = X_pred
                
            proba = model.predict_proba(X_input)[:, 1]
            ensemble_proba += weights[name] * proba
            
        # More aggressive non-linear scaling to emphasize higher probabilities
        # This will create a steeper curve for higher probabilities
        risk_scores = np.clip((np.power(ensemble_proba, 0.25) * 1300 - 300).astype(int), 0, 1000)
        
        features = {}
        for col in ['dti', 'int_rate', 'loan_amnt', 'annual_inc', 'fico_range_high', 'term']:
            if col in input_data.columns:
                features[col] = input_data[col].iloc[0]
        
        adjustments = []
        # DTI adjustment (higher DTI = higher risk, lower DTI = much safer)
        if 'dti' in features:
            if features['dti'] < 12:
                dti_adj = -150  # Strong reward for low DTI
            else:
                dti_adj = min(max((features['dti'] - 12) * 13, 0), 220)
            adjustments.append(dti_adj)
        # Interest rate adjustment (higher rate = higher risk, low rate = reward)
        if 'int_rate' in features:
            if features['int_rate'] < 8:
                rate_adj = -100
            else:
                rate_adj = min(max((features['int_rate'] - 8) * 12, 0), 170)
            adjustments.append(rate_adj)
        # Loan amount adjustment (higher loan = higher risk, low loan = reward)
        if 'loan_amnt' in features:
            if features['loan_amnt'] < 12000:
                loan_adj = -80
            else:
                loan_adj = min(max((features['loan_amnt'] - 12000) / 80, 0), 120)
            adjustments.append(loan_adj)
        # Income adjustment (higher income = lower risk)
        if 'annual_inc' in features:
            if features['annual_inc'] > 85000:
                inc_adj = -120
            else:
                inc_adj = min(max((100000 - features['annual_inc']) / 350, 0), 140)
            adjustments.append(inc_adj)
        # FICO score adjustment (higher score = lower risk, low score = strong penalty)
        if 'fico_range_high' in features:
            if features['fico_range_high'] > 780:
                fico_adj = -170
            else:
                fico_adj = min(max((700 - features['fico_range_high']) * 1.2, 0), 220)
            adjustments.append(fico_adj)
        # Term adjustment (longer term = higher risk)
        if 'term' in features:
            term_months = int(str(features['term']).split()[0]) if isinstance(features['term'], str) else features['term']
            term_adj = 70 if term_months > 36 else -30
            adjustments.append(term_adj)
        # Calculate final adjustment (average)
        if adjustments:
            risk_adjustment = sum(adjustments) / len(adjustments)
            risk_scores = np.clip(risk_scores + risk_adjustment, 0, 1000)
        # Clamp very strong profiles to max 300
        if all([
            features.get('fico_range_high', 0) > 780,
            features.get('dti', 100) < 10,
            features.get('annual_inc', 0) > 85000,
            features.get('loan_amnt', 99999) < 12000,
            features.get('int_rate', 100) < 8,
            features.get('term', '36 months') == '36 months'
        ]):
            risk_scores = np.clip(risk_scores, 0, 300)

        return risk_scores
        
    def generate_insights(self) -> None:
        """Generate business insights and recommendations"""
        print("\n=== BUSINESS INSIGHTS (NO DATA LEAKAGE) ===")

        # Feature importance insights
        if 'random_forest' in self.feature_importance:
            top_features = self.feature_importance['random_forest'].head(10)
            print("\nTop Risk Factors (Available at Application Time):")
            for idx, row in top_features.iterrows():
                print(f"- {row['feature']}: {row['importance']:.4f}")

        # Risk distribution insights
        if hasattr(self, 'y_test'):
            print(f"\nModel Performance:")
            print(f"- Overall default rate: {self.y_test.mean():.2%}")

        # Recommendations
        print(f"\n=== RECOMMENDATIONS ===")
        recommendations = [
            "Use only pre-approval features for production scoring",
            "Focus on top risk factors available at application time",
            "Set risk score thresholds based on business risk tolerance",
            "Monitor model performance on new originations",
            "Retrain model regularly with new data (using same pre-approval features)",
            "Consider additional pre-approval data sources if available"
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

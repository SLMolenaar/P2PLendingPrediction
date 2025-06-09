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

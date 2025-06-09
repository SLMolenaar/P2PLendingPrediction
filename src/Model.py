import os
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')


class Model:
    """Enhanced model class for loan default prediction with improved risk scoring"""

    def __init__(self, use_robust_scaling: bool = True, calibrate_models: bool = True):
        self.models = {}
        self.calibrated_models = {}
        self.feature_importance = {}
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.feature_names = None
        self.model_metrics = {}
        self.calibrate_models = calibrate_models
        self.risk_thresholds = None

    def set_data(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                 feature_names: List[str] = None):
        """Set training and test data with enhanced validation"""
        # Validate inputs
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Training and test sets must have the same number of features")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Enhanced scaling with outlier handling
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        print(
            f"[INFO] Data set successfully. Features: {len(self.feature_names)}, Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    def train_models(self) -> Dict:
        """Train multiple models with enhanced hyperparameters and calibration"""
        print("\n[INFO] Training enhanced models...")

        if self.X_train is None:
            raise ValueError("Data must be set first. Call set_data()")

        # Enhanced Logistic Regression with regularization
        lr = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=0.1,  # L2 regularization
            class_weight='balanced',  # Handle class imbalance
            solver='liblinear'
        )
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['logistic_regression'] = lr

        # Enhanced Random Forest with better hyperparameters
        rf = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=15,  # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf

        # Enhanced XGBoost with better hyperparameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
            # Handle imbalance
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model

        # Calibrate models for better probability estimates
        if self.calibrate_models:
            print("\n[INFO] Calibrating models for better probability estimates...")
            from tqdm import tqdm
            
            # Create a progress bar for model calibration
            model_names = list(self.models.keys())
            progress_bar = tqdm(
                model_names,
                desc="Calibrating models",
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                ncols=100
            )
            
            for name in progress_bar:
                progress_bar.set_description(f"Calibrating {name.replace('_', ' ').title()}")
                model = self.models[name]
                X_cal = self.X_train_scaled if name == 'logistic_regression' else self.X_train
                
                # Update progress bar with current model info
                progress_bar.set_postfix_str(f"Model: {name}")
                
                # Calibrate the model
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3, n_jobs=-1)
                calibrated.fit(X_cal, self.y_train)
                self.calibrated_models[name] = calibrated
            
            # Close the progress bar
            progress_bar.close()
            print("\n[INFO] Model calibration completed!")

        print(f"[INFO] Trained {len(self.models)} models")
        return self.models

    def evaluate_models(self, feature_names: List[str] = None) -> Dict:
        """Enhanced model evaluation with additional metrics"""
        print("\n=== ENHANCED MODEL EVALUATION ===")

        if not self.models:
            raise ValueError("Models must be trained first. Call train_models()")

        results = {}
        feature_names = feature_names or self.feature_names

        for name, model in self.models.items():
            print(f"\n--- {name.upper()} ---")

            # Use appropriate data
            X_test_eval = self.X_test_scaled if name == 'logistic_regression' else self.X_test

            # Get model to use (calibrated if available)
            eval_model = self.calibrated_models.get(name, model)

            # Predictions
            y_pred = eval_model.predict(X_test_eval)
            y_pred_proba = eval_model.predict_proba(X_test_eval)[:, 1]

            # Enhanced metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)

            # Cross-validation score on training data
            X_cv = self.X_train_scaled if name == 'logistic_regression' else self.X_train
            cv_scores = cross_val_score(model, X_cv, self.y_train, cv=5, scoring='roc_auc')

            print(f"AUC Score: {auc_score:.4f}")
            print(f"CV AUC Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))

            # Store comprehensive results
            results[name] = {
                'auc': auc_score,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            self.model_metrics[name] = results[name]

            # Enhanced feature importance analysis
            if hasattr(model, 'feature_importances_') and feature_names:
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                self.feature_importance[name] = importance
                print(f"\nTop 10 Most Important Features:")
                for idx, row in importance.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")

        return results

    def _calculate_optimal_thresholds(self, probabilities: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """Calculate optimal risk score thresholds based on model performance"""
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(actuals, probabilities)

        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(actuals, probabilities)

        # Find optimal threshold using Youden's index
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = roc_thresholds[optimal_idx]

        # Convert to risk score scale (0-1000)
        risk_thresholds = {
            'very_low': int(optimal_threshold * 0.3 * 1000),
            'low': int(optimal_threshold * 0.6 * 1000),
            'medium': int(optimal_threshold * 0.9 * 1000),
            'high': int(optimal_threshold * 1.2 * 1000),
            'very_high': 1000
        }

        return risk_thresholds

    def create_risk_score(self, method: str = 'ensemble') -> pd.DataFrame:
        """Enhanced risk scoring system with dynamic thresholds"""
        print(f"\n[INFO] Creating enhanced risk scoring system using {method} method...")

        if not self.models:
            raise ValueError("Models must be trained first. Call train_models()")

        if method == 'ensemble':
            # Dynamic ensemble weights based on model performance
            total_auc = sum(self.model_metrics[name]['auc'] for name in self.models.keys())
            weights = {name: self.model_metrics[name]['auc'] / total_auc for name in self.models.keys()}

            print(f"Dynamic ensemble weights: {weights}")

            ensemble_proba = np.zeros(len(self.y_test))
            for name, model in self.models.items():
                eval_model = self.calibrated_models.get(name, model)
                X_test_eval = self.X_test_scaled if name == 'logistic_regression' else self.X_test
                proba = eval_model.predict_proba(X_test_eval)[:, 1]
                ensemble_proba += weights[name] * proba

            # Enhanced risk score calculation with non-linear transformation
            risk_scores = self._transform_probabilities_to_risk_scores(ensemble_proba)

        else:
            # Use best performing model
            best_model_name = max(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['auc'])
            print(f"Using best model: {best_model_name} (AUC: {self.model_metrics[best_model_name]['auc']:.4f})")

            best_model = self.calibrated_models.get(best_model_name, self.models[best_model_name])
            X_test_eval = self.X_test_scaled if best_model_name == 'logistic_regression' else self.X_test
            ensemble_proba = best_model.predict_proba(X_test_eval)[:, 1]
            risk_scores = self._transform_probabilities_to_risk_scores(ensemble_proba)

        # Calculate optimal thresholds
        self.risk_thresholds = self._calculate_optimal_thresholds(ensemble_proba, self.y_test)

        # Create dynamic risk categories
        risk_categories = pd.cut(risk_scores,
                                 bins=[0, self.risk_thresholds['very_low'],
                                       self.risk_thresholds['low'],
                                       self.risk_thresholds['medium'],
                                       self.risk_thresholds['high'], 1000],
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Enhanced results summary
        results_df = pd.DataFrame({
            'actual_default': self.y_test.values,
            'risk_score': risk_scores,
            'risk_category': risk_categories,
            'ensemble_probability': ensemble_proba,
            'risk_percentile': pd.qcut(risk_scores, q=10, labels=False) + 1
        })

        # Comprehensive analysis
        print(f"\nRisk Score Distribution:")
        print(results_df['risk_category'].value_counts().sort_index())

        print(f"\nDefault Rate by Risk Category:")
        default_by_risk = results_df.groupby('risk_category')['actual_default'].agg(['mean', 'count'])
        print(default_by_risk)

        print(f"\nRisk Score Statistics:")
        print(results_df['risk_score'].describe())

        return results_df

    def _transform_probabilities_to_risk_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Enhanced probability to risk score transformation"""
        # Multi-stage transformation for better score distribution

        # Stage 1: Square root transformation to spread out lower probabilities
        transformed = np.sqrt(probabilities)

        # Stage 2: Apply sigmoid-like transformation for smooth scaling
        scaled = 1 / (1 + np.exp(-10 * (transformed - 0.5)))

        # Stage 3: Final scaling to 0-1000 range with floor at 0
        risk_scores = np.clip((scaled * 1000).astype(int), 0, 1000)

        return risk_scores

    def predict_risk(self, input_data: pd.DataFrame) -> np.ndarray:
        """Enhanced risk prediction with comprehensive feature engineering
        
        Args:
            input_data: DataFrame containing the input features with original feature names
            
        Returns:
            np.ndarray: Array of risk scores
        """
        if not self.models:
            raise ValueError("Models must be trained first. Call train_models()")

        if self.feature_names is None:
            raise ValueError("Feature names not found. The model may not have been properly trained.")
            
        # Make a copy of input data to avoid modifying the original
        input_data = input_data.copy()
        
        # If feature names are generic (feature_0, feature_1, etc.), we need to map them
        if all(f.startswith('feature_') for f in self.feature_names):
            # Create a mapping from original feature names to generic feature names
            # This assumes the order of features in the input matches the training order
            if len(input_data.columns) != len(self.feature_names):
                raise ValueError(f"Number of input features ({len(input_data.columns)}) "
                               f"does not match number of training features ({len(self.feature_names)})")
            
            # Rename columns to match the generic feature names used during training
            input_data.columns = self.feature_names[:len(input_data.columns)]
        
        # Ensure all required features are present in input_data
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            # If we have some features but not all, try to map them by position
            if len(input_data.columns) == len(self.feature_names):
                input_data.columns = self.feature_names
                missing_features = set()
            
            if missing_features:
                raise ValueError(f"Missing required features in input data: {missing_features}")
        
        # Keep only the features that were used during training
        input_data = input_data[self.feature_names]

        # Enhanced feature preparation
        X_pred = self._prepare_prediction_features(input_data)
        
        # Ensure the features are in the same order as during training
        X_pred = X_pred[self.feature_names]

        # Get ensemble predictions using calibrated models if available
        ensemble_proba = self._get_ensemble_predictions(X_pred)

        # Transform to risk scores
        base_risk_scores = self._transform_probabilities_to_risk_scores(ensemble_proba)
        
        # Apply enhanced business logic adjustments
        adjusted_risk_scores = self._apply_business_adjustments(base_risk_scores, input_data)

        return adjusted_risk_scores

    def _prepare_prediction_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature preparation with better defaults"""
        X_pred = pd.DataFrame(index=input_data.index)

        # Feature defaults based on statistical analysis
        feature_defaults = {
            'loan_amnt': 15000, 'funded_amnt': 15000, 'funded_amnt_inv': 15000,
            'int_rate': 12.0, 'term': 36, 'annual_inc': 65000, 'dti': 18.0,
            'fico_range_low': 680, 'fico_range_high': 684,
            'revol_util': 35.0, 'bc_util': 30.0, 'il_util': 25.0, 'all_util': 32.0,
            'open_acc': 11, 'total_acc': 25, 'pub_rec': 0, 'delinq_2yrs': 0, 'inq_last_6mths': 1
        }

        for feature in self.feature_names:
            if feature in input_data.columns:
                X_pred[feature] = input_data[feature]
            else:
                # Enhanced default logic
                if feature in feature_defaults:
                    X_pred[feature] = feature_defaults[feature]
                elif 'fico' in feature.lower():
                    X_pred[feature] = 680
                elif any(util in feature for util in ['util', 'ratio']):
                    X_pred[feature] = 30.0
                elif any(acc in feature for acc in ['acc', 'account']):
                    X_pred[feature] = 10
                else:
                    X_pred[feature] = 0

        # Enhanced data cleaning
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce')

        # Fill NaN values with column-specific defaults
        for col in X_pred.columns:
            if X_pred[col].isna().any():
                if col in feature_defaults:
                    X_pred[col].fillna(feature_defaults[col], inplace=True)
                else:
                    X_pred[col].fillna(X_pred[col].median() or 0, inplace=True)

        return X_pred[self.feature_names]

    def _get_ensemble_predictions(self, X_pred: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions using best available models"""
        ensemble_proba = np.zeros(len(X_pred))

        # Use dynamic weights if available, otherwise equal weights
        if hasattr(self, 'model_metrics') and self.model_metrics:
            total_auc = sum(self.model_metrics[name]['auc'] for name in self.models.keys())
            weights = {name: self.model_metrics[name]['auc'] / total_auc for name in self.models.keys()}
        else:
            weights = {'logistic_regression': 0.3, 'random_forest': 0.3, 'xgboost': 0.4}

        for name, model in self.models.items():
            # Use calibrated model if available
            eval_model = self.calibrated_models.get(name, model)

            # Prepare input data
            if name == 'logistic_regression':
                X_input = self.scaler.transform(X_pred)
            else:
                X_input = X_pred

            proba = eval_model.predict_proba(X_input)[:, 1]
            ensemble_proba += weights[name] * proba

        return ensemble_proba

    def _apply_business_adjustments(self, base_scores: np.ndarray, input_data: pd.DataFrame) -> np.ndarray:
        """Enhanced business logic adjustments with more sophisticated rules"""
        adjusted_scores = base_scores.copy()

        for idx in range(len(input_data)):
            row = input_data.iloc[idx]
            adjustments = []

            # Enhanced DTI adjustment with non-linear scaling
            if 'dti' in row:
                dti = float(row['dti'])
                if dti < 8:
                    dti_adj = -200  # Excellent DTI
                elif dti < 15:
                    dti_adj = -100  # Good DTI
                elif dti < 25:
                    dti_adj = 0  # Average DTI
                else:
                    dti_adj = min((dti - 25) * 15, 300)  # High DTI penalty
                adjustments.append(dti_adj)

            # Enhanced interest rate adjustment
            if 'int_rate' in row:
                rate = float(row['int_rate'])
                if rate < 7:
                    rate_adj = -150  # Excellent rate
                elif rate < 12:
                    rate_adj = -50  # Good rate
                elif rate < 18:
                    rate_adj = (rate - 12) * 20  # Moderate penalty
                else:
                    rate_adj = min((rate - 12) * 25, 250)  # High rate penalty
                adjustments.append(rate_adj)

            # Enhanced loan amount adjustment (relative to income)
            if 'loan_amnt' in row and 'annual_inc' in row:
                loan_to_income = float(row['loan_amnt']) / max(float(row['annual_inc']), 1000)
                if loan_to_income < 0.15:
                    loan_adj = -100
                elif loan_to_income < 0.3:
                    loan_adj = -50
                elif loan_to_income < 0.5:
                    loan_adj = 0
                else:
                    loan_adj = min((loan_to_income - 0.5) * 400, 200)
                adjustments.append(loan_adj)

            # Enhanced income adjustment with percentile-based scoring
            if 'annual_inc' in row:
                income = float(row['annual_inc'])
                if income > 150000:
                    inc_adj = -150  # High income
                elif income > 100000:
                    inc_adj = -100  # Good income
                elif income > 70000:
                    inc_adj = -50  # Moderate income
                elif income > 40000:
                    inc_adj = 0  # Baseline
                else:
                    inc_adj = min((50000 - income) / 200, 150)  # Low income penalty
                adjustments.append(inc_adj)

            # Enhanced FICO adjustment with credit tier system
            fico_col = 'fico_range_high' if 'fico_range_high' in row else 'fico_range_low'
            if fico_col in row:
                fico = float(row[fico_col])
                if fico >= 800:
                    fico_adj = -250  # Exceptional credit
                elif fico >= 740:
                    fico_adj = -150  # Very good credit
                elif fico >= 670:
                    fico_adj = -75  # Good credit
                elif fico >= 580:
                    fico_adj = (670 - fico) * 2  # Fair credit
                else:
                    fico_adj = min((670 - fico) * 3, 350)  # Poor credit
                adjustments.append(fico_adj)

            # Enhanced term adjustment
            if 'term' in row:
                term_val = row['term']
                if isinstance(term_val, str):
                    term_months = int(term_val.split()[0])
                else:
                    term_months = int(term_val)

                if term_months <= 36:
                    term_adj = -50  # Shorter term is better
                else:
                    term_adj = (term_months - 36) * 3  # Penalty for longer terms
                adjustments.append(term_adj)

            # Calculate weighted adjustment
            if adjustments:
                final_adjustment = np.mean(adjustments)
                adjusted_scores[idx] = max(0, min(1000, adjusted_scores[idx] + final_adjustment))

            # Apply tier-based capping for exceptional profiles
            exceptional_profile = all([
                row.get('fico_range_high', 0) > 780,
                row.get('dti', 100) < 12,
                row.get('annual_inc', 0) > 100000,
                row.get('int_rate', 100) < 8
            ])

            if exceptional_profile:
                adjusted_scores[idx] = min(adjusted_scores[idx], 250)  # Cap exceptional profiles

        return adjusted_scores.astype(int)

    def generate_insights(self) -> None:
        """Enhanced business insights with actionable recommendations"""
        print("\n=== ENHANCED BUSINESS INSIGHTS ===")

        # Model performance comparison
        if self.model_metrics:
            print(f"\nModel Performance Comparison:")
            for name, metrics in self.model_metrics.items():
                print(
                    f"- {name}: AUC = {metrics['auc']:.4f}, CV AUC = {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        # Feature importance insights across models
        if self.feature_importance:
            print(f"\nTop Risk Factors (Consensus across models):")
            # Combine feature importance from all models
            all_features = {}
            for model_name, importance_df in self.feature_importance.items():
                for _, row in importance_df.iterrows():
                    feature = row['feature']
                    importance = row['importance']
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)

            # Calculate average importance
            avg_importance = {feature: np.mean(scores) for feature, scores in all_features.items()}
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            for feature, importance in sorted_features[:10]:
                print(f"- {feature}: {importance:.4f}")

        # Risk threshold insights
        if self.risk_thresholds:
            print(f"\nOptimal Risk Score Thresholds:")
            for category, threshold in self.risk_thresholds.items():
                print(f"- {category.replace('_', ' ').title()}: {threshold}")

        # Enhanced recommendations
        print(f"\n=== ENHANCED RECOMMENDATIONS ===")
        recommendations = [
            "Deploy ensemble model with calibrated probabilities for best performance",
            "Use dynamic risk thresholds based on portfolio performance",
            "Focus underwriting on top consensus risk factors identified across models",
            "Implement automated model monitoring with performance alerts",
            "Consider A/B testing different risk score cutoffs for business optimization",
            "Retrain models quarterly with performance-based feature selection",
            "Implement model explainability for regulatory compliance",
            "Consider external data sources to improve low-performing features",
            "Set up automated data quality monitoring for input features",
            "Establish model governance framework for production deployment"
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")

    def save_model(self, filepath='../saved_models/trained_model.joblib') -> None:
        """Save the complete model pipeline"""
        model_data = {
            'models': self.models,
            'calibrated_models': self.calibrated_models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_metrics': self.model_metrics,
            'risk_thresholds': self.risk_thresholds,
            'created_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        print(f"[INFO] Model saved to {filepath}")

    def load_model(self, filepath='../saved_models/trained_model.joblib') -> None:
        """Load a saved model pipeline"""
        # If we get here, load the existing model
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.calibrated_models = model_data.get('calibrated_models', {})
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance', {})
        self.model_metrics = model_data.get('model_metrics', {})
        self.risk_thresholds = model_data.get('risk_thresholds', None)
        print(f"[INFO] Model loaded from {filepath}")
        print(f"[INFO] Model created at: {model_data.get('created_at', 'Unknown')}")
        return True

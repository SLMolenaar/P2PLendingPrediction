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
        self.X_train_df = None
        self.X_test_df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.feature_names = None
        self.model_metrics = {}
        self.calibrate_models = calibrate_models
        self.risk_thresholds = None

    def set_data(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                 feature_names: List[str] = None):
        """Set training and test data with enhanced validation and feature name preservation"""
        # Validate inputs
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Training and test sets must have the same number of features")

        # Convert to pandas DataFrames to preserve feature names
        if feature_names is not None:
            self.feature_names = feature_names
            # Ensure we have the right number of feature names
            if len(feature_names) != X_train.shape[1]:
                print(
                    f"[WARNING] Feature names length ({len(feature_names)}) doesn't match data features ({X_train.shape[1]})")
                # Use provided names for available features, generate for missing ones
                complete_names = feature_names[:X_train.shape[1]]
                if len(complete_names) < X_train.shape[1]:
                    complete_names.extend([f'feature_{i}' for i in range(len(complete_names), X_train.shape[1])])
                self.feature_names = complete_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Convert to DataFrames with proper column names
        self.X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        self.X_test_df = pd.DataFrame(X_test, columns=self.feature_names)

        # Keep numpy arrays for backward compatibility
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Enhanced scaling with outlier handling
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        print(
            f"[INFO] Data set successfully. Features: {len(self.feature_names)}, Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"[INFO] Feature names: {self.feature_names[:5]}{'...' if len(self.feature_names) > 5 else ''}")

    def train_models(self) -> Dict:
        """Train multiple models with enhanced hyperparameters and proper feature name handling"""
        print("\n[INFO] Training enhanced models with proper feature names...")

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
        # Train with DataFrame to preserve feature names
        lr.fit(self.X_train_df, self.y_train)
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
        # Train with DataFrame to preserve feature names
        rf.fit(self.X_train_df, self.y_train)
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
        )
        # Train with DataFrame to preserve feature names
        xgb_model.fit(self.X_train_df, self.y_train)
        self.models['xgboost'] = xgb_model

        # Calibrate models for better probability estimates
        if self.calibrate_models:
            print("\n[INFO] Calibrating models for better probability estimates...")
            for name, model in self.models.items():
                print(f"Calibrating {name}...")
                # Use DataFrame for calibration as well
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3, n_jobs=-1)
                calibrated.fit(self.X_train_df, self.y_train)
                self.calibrated_models[name] = calibrated

        print(f"[INFO] Trained {len(self.models)} models with proper feature names")
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

            # Use DataFrame for evaluation
            eval_model = self.calibrated_models.get(name, model)

            # Predictions using DataFrame
            y_pred = eval_model.predict(self.X_test_df)
            y_pred_proba = eval_model.predict_proba(self.X_test_df)[:, 1]

            # Enhanced metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)

            # Cross-validation score on training data using DataFrame
            cv_scores = cross_val_score(model, self.X_train_df, self.y_train, cv=2, scoring='roc_auc')

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
                proba = eval_model.predict_proba(self.X_test_df)[:, 1]
                ensemble_proba += weights[name] * proba

            # Enhanced risk score calculation with non-linear transformation
            risk_scores = self._transform_probabilities_to_risk_scores(ensemble_proba)

        else:
            # Use best performing model
            best_model_name = max(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['auc'])
            print(f"Using best model: {best_model_name} (AUC: {self.model_metrics[best_model_name]['auc']:.4f})")

            best_model = self.calibrated_models.get(best_model_name, self.models[best_model_name])
            ensemble_proba = best_model.predict_proba(self.X_test_df)[:, 1]
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
        """Enhanced risk prediction with proper feature name handling"""
        if not self.models:
            raise ValueError("Models must be trained first. Call train_models()")

        if self.feature_names is None:
            raise ValueError("Feature names not found. The model may not have been properly trained.")

        print(f"[DEBUG] Expected features: {len(self.feature_names)}")
        print(f"[DEBUG] Input columns: {list(input_data.columns)}")

        # Create a DataFrame with all expected features, initialized with defaults
        X_pred = pd.DataFrame(index=input_data.index, columns=self.feature_names)

        # Fill with defaults first
        feature_defaults = {
            'loan_amnt': 15000, 'funded_amnt': 15000, 'funded_amnt_inv': 15000,
            'int_rate': 12.0, 'term': 36, 'annual_inc': 65000, 'dti': 18.0,
            'fico_range_low': 680, 'fico_range_high': 684,
            'revol_util': 35.0, 'bc_util': 30.0, 'il_util': 25.0, 'all_util': 32.0,
            'open_acc': 11, 'total_acc': 25, 'pub_rec': 0, 'delinq_2yrs': 0,
            'inq_last_6mths': 1, 'installment': 400, 'revol_bal': 15000,
            'acc_now_delinq': 0, 'collections_12_mths_ex_med': 0, 'policy_code': 1,
            'mths_since_last_delinq': 30, 'mths_since_last_record': 60, 'initial_list_status': 0,
            'application_type': 0, 'verification_status': 0, 'home_ownership': 0,
            'purpose': 0, 'addr_state': 0, 'grade': 0, 'sub_grade': 0,
            'emp_length': 0, 'zip_code': 0, 'emp_length_numeric': 5,
            'employment_stability': 1, 'debt_to_income_ratio': 0.2,
            'income_risk_score': 1.0, 'loan_to_income': 0.2, 'high_loan_to_income': 0,
            'credit_utilization_risk': 1.0, 'high_risk_purpose': 0,
            'fico_range_avg': 680, 'fico_risk_category': 2.0,
            'high_recent_inquiries': 0, 'has_recent_delinquency': 0,
            'composite_risk_score': 5.0
        }

        # Apply defaults
        for feature in self.feature_names:
            if feature in feature_defaults:
                X_pred[feature] = feature_defaults[feature]
            elif 'fico' in feature.lower():
                X_pred[feature] = 680
            elif any(keyword in feature.lower() for keyword in ['util', 'ratio']):
                X_pred[feature] = 30.0
            elif any(keyword in feature.lower() for keyword in ['acc', 'account']):
                X_pred[feature] = 10
            elif any(keyword in feature.lower() for keyword in ['mths', 'month']):
                X_pred[feature] = 30
            elif any(keyword in feature.lower() for keyword in ['amt', 'bal', 'amount']):
                X_pred[feature] = 5000
            elif 'rate' in feature.lower():
                X_pred[feature] = 12.0
            elif any(keyword in feature.lower() for keyword in ['delinq', 'pub_rec', 'inq']):
                X_pred[feature] = 0
            else:
                X_pred[feature] = 0

        # Override with actual input values where available
        for col in input_data.columns:
            if col in self.feature_names:
                X_pred[col] = input_data[col]

        # Handle term conversion if needed
        if 'term' in X_pred.columns and 'term' in input_data.columns:
            term_val = input_data['term'].iloc[0] if len(input_data) > 0 else "36 months"
            if isinstance(term_val, str):
                term_numeric = int(term_val.split()[0])
                X_pred['term'] = term_numeric

        # Apply feature engineering
        X_pred = self._apply_feature_engineering_for_prediction(X_pred)

        # Ensure proper data types and handle missing values
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce')
        X_pred = X_pred.fillna(0)

        # Ensure we have exactly the features the model expects, in the right order
        X_pred = X_pred[self.feature_names]

        print(f"[DEBUG] Final prediction data shape: {X_pred.shape}")
        print(f"[DEBUG] Final columns: {list(X_pred.columns)[:5]}...")

        # Get ensemble predictions using the DataFrame
        ensemble_proba = self._get_ensemble_predictions_dataframe(X_pred)

        # Transform to risk scores
        base_risk_scores = self._transform_probabilities_to_risk_scores(ensemble_proba)

        # Apply enhanced business logic adjustments
        adjusted_risk_scores = self._apply_business_adjustments(base_risk_scores, input_data)

        return adjusted_risk_scores

    def _get_ensemble_predictions_dataframe(self, X_pred: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions using DataFrame to preserve feature names"""
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

            # Use DataFrame for prediction to preserve feature names
            proba = eval_model.predict_proba(X_pred)[:, 1]
            ensemble_proba += weights[name] * proba

        return ensemble_proba

    def _apply_feature_engineering_for_prediction(self, X_pred: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering for prediction data"""

        # Create debt to income ratio if missing
        if 'debt_to_income_ratio' in self.feature_names and 'debt_to_income_ratio' not in X_pred.columns:
            if 'loan_amnt' in X_pred.columns and 'annual_inc' in X_pred.columns:
                X_pred['debt_to_income_ratio'] = X_pred['loan_amnt'] / (X_pred['annual_inc'] + 1)

        # Create income risk score if missing
        if 'income_risk_score' in self.feature_names and 'income_risk_score' not in X_pred.columns:
            if 'annual_inc' in X_pred.columns:
                X_pred['income_risk_score'] = pd.cut(X_pred['annual_inc'],
                                                     bins=[0, 30000, 60000, 100000, float('inf')],
                                                     labels=[3, 2, 1, 0]).astype(float)

        # Create loan to income if missing
        if 'loan_to_income' in self.feature_names and 'loan_to_income' not in X_pred.columns:
            if 'loan_amnt' in X_pred.columns and 'annual_inc' in X_pred.columns:
                X_pred['loan_to_income'] = X_pred['loan_amnt'] / (X_pred['annual_inc'] + 1)

        # Create high loan to income flag if missing
        if 'high_loan_to_income' in self.feature_names and 'high_loan_to_income' not in X_pred.columns:
            if 'loan_to_income' in X_pred.columns:
                X_pred['high_loan_to_income'] = (X_pred['loan_to_income'] > 0.3).astype(int)

        # Create employment length numeric if missing
        if 'emp_length_numeric' in self.feature_names and 'emp_length_numeric' not in X_pred.columns:
            X_pred['emp_length_numeric'] = 5  # Default to 5 years experience

        # Create employment stability if missing
        if 'employment_stability' in self.feature_names and 'employment_stability' not in X_pred.columns:
            X_pred['employment_stability'] = 1  # Default to stable

        # Create credit utilization risk if missing
        if 'credit_utilization_risk' in self.feature_names and 'credit_utilization_risk' not in X_pred.columns:
            if 'revol_util' in X_pred.columns:
                X_pred['credit_utilization_risk'] = pd.cut(X_pred['revol_util'],
                                                           bins=[0, 30, 60, 80, 100],
                                                           labels=[0, 1, 2, 3]).astype(float)

        # Create high risk purpose if missing
        if 'high_risk_purpose' in self.feature_names and 'high_risk_purpose' not in X_pred.columns:
            X_pred['high_risk_purpose'] = 0  # Default to low risk purpose

        # Create FICO average if missing
        if 'fico_range_avg' in self.feature_names and 'fico_range_avg' not in X_pred.columns:
            if 'fico_range_low' in X_pred.columns and 'fico_range_high' in X_pred.columns:
                X_pred['fico_range_avg'] = (X_pred['fico_range_low'] + X_pred['fico_range_high']) / 2

        # Create FICO risk category if missing
        if 'fico_risk_category' in self.feature_names and 'fico_risk_category' not in X_pred.columns:
            if 'fico_range_avg' in X_pred.columns:
                X_pred['fico_risk_category'] = pd.cut(X_pred['fico_range_avg'],
                                                      bins=[0, 600, 650, 700, 750, 850],
                                                      labels=[4, 3, 2, 1, 0]).astype(float)

        # Create high recent inquiries if missing
        if 'high_recent_inquiries' in self.feature_names and 'high_recent_inquiries' not in X_pred.columns:
            if 'inq_last_6mths' in X_pred.columns:
                X_pred['high_recent_inquiries'] = (X_pred['inq_last_6mths'] > 2).astype(int)

        # Create has recent delinquency if missing
        if 'has_recent_delinquency' in self.feature_names and 'has_recent_delinquency' not in X_pred.columns:
            if 'delinq_2yrs' in X_pred.columns:
                X_pred['has_recent_delinquency'] = (X_pred['delinq_2yrs'] > 0).astype(int)

        # Create composite risk score if missing
        if 'composite_risk_score' in self.feature_names and 'composite_risk_score' not in X_pred.columns:
            risk_features = ['debt_to_income_ratio', 'credit_utilization_risk', 'high_risk_purpose',
                             'fico_risk_category', 'high_loan_to_income', 'high_recent_inquiries',
                             'has_recent_delinquency']
            available_risk_features = [f for f in risk_features if f in X_pred.columns]

            if available_risk_features:
                X_pred['composite_risk_score'] = X_pred[available_risk_features].sum(axis=1)
            else:
                X_pred['composite_risk_score'] = 5.0  # Default moderate risk

        return X_pred

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
                    term_adj = (term_months - 36) * 3  #
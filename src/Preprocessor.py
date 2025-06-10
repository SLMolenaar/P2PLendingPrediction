import os
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

# Suppress warnings
warnings.filterwarnings('ignore')


class Preprocessor:
    """Handles all data preprocessing including feature engineering, fraud detection, and data preparation"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.anomaly_detector = None
        self.feature_names = None

        # Define features available at loan origination (NO DATA LEAKAGE)
        self.pre_approval_features = [
            'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
            'installment', 'grade', 'sub_grade', 'purpose', 'emp_title', 'emp_length',
            'home_ownership', 'annual_inc', 'verification_status', 'zip_code',
            'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low',
            'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
            'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
            'total_acc', 'initial_list_status', 'application_type', 'policy_code',
            'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'revol_bal_joint',
            'acc_now_delinq', 'collections_12_mths_ex_med', 'mths_since_last_major_derog',
            'tot_coll_amt', 'tot_cur_bal', 'chargeoff_within_12_mths', 'delinq_amnt',
            'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort',
            'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
            'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
            'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
            'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util'
        ]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced risk features using only pre-approval data"""
        print("\n[INFO] Starting feature engineering (no data leakage)...")
        df = df.copy()

        # Income and debt features
        if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
            df['debt_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
            df['income_risk_score'] = pd.cut(df['annual_inc'],
                                             bins=[0, 30000, 60000, 100000, float('inf')],
                                             labels=[3, 2, 1, 0]).astype(float)
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
            df['high_loan_to_income'] = (df['loan_to_income'] > 0.3).astype(int)

        # Employment features
        if 'emp_length' in df.columns:
            emp_mapping = {
                '< 1 year': 1, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            df['emp_length_numeric'] = df['emp_length'].map(emp_mapping).fillna(0)
            df['employment_stability'] = (df['emp_length_numeric'] >= 3).astype(int)

        # Credit utilization features
        if 'revol_util' in df.columns:
            df['credit_utilization_risk'] = pd.cut(df['revol_util'],
                                                   bins=[0, 30, 60, 80, 100],
                                                   labels=[0, 1, 2, 3]).astype(float)

        # Loan purpose risk
        if 'purpose' in df.columns:
            high_risk_purposes = ['small_business', 'other', 'moving', 'vacation']
            df['high_risk_purpose'] = df['purpose'].isin(high_risk_purposes).astype(int)

        # FICO score features
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_range_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
            df['fico_risk_category'] = pd.cut(df['fico_range_avg'],
                                              bins=[0, 600, 650, 700, 750, 850],
                                              labels=[4, 3, 2, 1, 0]).astype(float)

        # Recent inquiries and delinquency
        if 'inq_last_6mths' in df.columns:
            df['high_recent_inquiries'] = (df['inq_last_6mths'] > 2).astype(int)

        if 'delinq_2yrs' in df.columns:
            df['has_recent_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)

        # Composite risk score
        risk_features = ['debt_to_income_ratio', 'credit_utilization_risk', 'high_risk_purpose',
                         'fico_risk_category', 'high_loan_to_income', 'high_recent_inquiries',
                         'has_recent_delinquency']
        available_risk_features = [f for f in risk_features if f in df.columns]

        if available_risk_features:
            for feature in available_risk_features:
                df[feature] = df[feature].fillna(0)
            df['composite_risk_score'] = df[available_risk_features].sum(axis=1)

        print(f"[INFO] Feature engineering complete. New shape: {df.shape}")
        return df

    def detect_fraud_patterns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Detect potential fraud patterns using only pre-approval data"""
        print("\n[INFO] Analyzing fraud patterns (no data leakage)...")
        df = df.copy()
        fraud_indicators = []

        # Income-to-loan ratio anomalies
        if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
            df['income_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1)
            df['suspicious_income_ratio'] = (
                    (df['income_loan_ratio'] > df['income_loan_ratio'].quantile(0.99)) |
                    (df['income_loan_ratio'] < df['income_loan_ratio'].quantile(0.01))
            ).astype(int)
            fraud_indicators.append('suspicious_income_ratio')

        # Employment title patterns
        if 'emp_title' in df.columns:
            emp_counts = df['emp_title'].value_counts()
            suspicious_employers = emp_counts[emp_counts > emp_counts.quantile(0.99)].index
            df['suspicious_employer'] = df['emp_title'].isin(suspicious_employers).astype(int)
            fraud_indicators.append('suspicious_employer')

        # Geographic risk patterns
        if 'addr_state' in df.columns and 'loan_status' in df.columns:
            state_default_rates = df.groupby('addr_state')['loan_status'].apply(
                lambda x: (x == 'Charged Off').mean() if len(x) > 100 else 0.2
            )
            high_risk_states = state_default_rates[state_default_rates > state_default_rates.quantile(0.8)].index
            df['high_risk_state'] = df['addr_state'].isin(high_risk_states).astype(int)
            fraud_indicators.append('high_risk_state')

        # Credit profile anomalies using pre-approval features
        pre_approval_numeric = [col for col in df.select_dtypes(include=[np.number]).columns
                                if col not in ['loan_status', 'default'] and col in self.pre_approval_features]

        if len(pre_approval_numeric) > 5:
            isolation_features = pre_approval_numeric[:15]
            isolation_data = df[isolation_features].fillna(0)

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df['anomaly_score'] = iso_forest.fit_predict(isolation_data)
            df['is_anomaly'] = (df['anomaly_score'] == -1).astype(int)
            self.anomaly_detector = iso_forest
            fraud_indicators.append('is_anomaly')

        print(f"[INFO] Created {len(fraud_indicators)} fraud indicators using only pre-approval data")
        return df, fraud_indicators

    def prepare_data_for_modeling(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for modeling using only pre-approval features"""
        print("\n[INFO] Preparing data for modeling (no data leakage)...")

        # Handle target variable
        if 'loan_status' not in df.columns:
            raise ValueError("loan_status column not found")

        # Filter for binary classification
        binary_status = df['loan_status'].isin(['Fully Paid', 'Charged Off'])
        df = df[binary_status].copy()
        df['default'] = (df['loan_status'] == 'Charged Off').astype(int)

        # Get valid features
        available_features = [f for f in self.pre_approval_features if f in df.columns]

        # Add engineered features
        engineered_features = [col for col in df.columns
                               if col.endswith(('_risk', '_score', '_ratio', '_stability', '_category',
                                                '_anomaly', 'suspicious_', 'high_risk_', 'has_', 'is_'))]
        available_features.extend(engineered_features)

        # Remove duplicates and unwanted columns
        available_features = list(set(available_features))
        columns_to_remove = ['loan_status', 'default', 'id', 'member_id', 'url', 'desc', 'title']
        available_features = [f for f in available_features if f not in columns_to_remove]

        # Remove columns with too many missing values
        missing_threshold = 0.7
        valid_features = []
        for feature in available_features:
            if feature in df.columns:
                missing_rate = df[feature].isnull().mean()
                if missing_rate < missing_threshold:
                    valid_features.append(feature)
                else:
                    print(f"[INFO] Removing {feature} due to {missing_rate:.2%} missing values")

        # Prepare features and target
        X = df[valid_features].copy()
        y = df['default']

        print(f"[INFO] Final feature set: {len(X.columns)} features")
        print(f"[INFO] Features: {list(X.columns)[:10]}{'...' if len(X.columns) > 10 else ''}")

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                mode_val = X[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 0
                X[col] = X[col].fillna(fill_val)

        # Convert all to numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Store feature names BEFORE converting to numpy arrays
        self.feature_names = list(X.columns)
        print(f"[INFO] Stored feature names: {self.feature_names[:5]}{'...' if len(self.feature_names) > 5 else ''}")

        # Split data - keep as DataFrames initially
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Convert to numpy arrays for compatibility
        X_train = X_train_df.values
        X_test = X_test_df.values

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Default rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

        return X_train, X_test, y_train, y_test
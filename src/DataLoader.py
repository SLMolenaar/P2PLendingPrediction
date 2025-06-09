import os
import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles data loading, exploration, and initial data quality checks"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

        # Features that cause DATA LEAKAGE (available only after loan origination)
        self.post_origination_features = [
            'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
            'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
            'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
            'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
            'last_fico_range_high', 'last_fico_range_low', 'hardship_flag',
            'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term',
            'hardship_amount', 'hardship_start_date', 'hardship_end_date',
            'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
            'hardship_loan_status', 'orig_projected_additional_accrued_interest',
            'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
            'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
            'settlement_date', 'settlement_amount', 'settlement_percentage',
            'settlement_term', 'disbursement_method', 'pymnt_plan', 'issue_d'
        ]

    def load_data(self) -> pd.DataFrame:
        """Load CSV data from file"""
        print("[INFO] Loading dataset...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        print(f"Dataset shape: {self.df.shape}")
        return self.df

    def explore_data(self) -> None:
        """Perform comprehensive data exploration"""
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")

        print("\n=== DATASET OVERVIEW ===")
        print(f"Columns: {len(self.df.columns)}")
        print(f"Rows: {len(self.df)}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Target variable analysis
        if 'loan_status' in self.df.columns:
            print(f"\nLoan Status Distribution:")
            print(self.df['loan_status'].value_counts())

    def remove_data_leakage(self) -> pd.DataFrame:
        """Remove features that cause data leakage"""
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")

        print(
            f"\n[INFO] Removing {len(self.post_origination_features)} post-origination features to prevent data leakage...")
        leakage_features_present = [f for f in self.post_origination_features if f in self.df.columns]
        print(f"[INFO] Found {len(leakage_features_present)} leakage features in dataset")

        if leakage_features_present:
            self.df = self.df.drop(columns=leakage_features_present)
            print(f"[INFO] Dataset shape after removing leakage features: {self.df.shape}")

        return self.df

    def get_data(self) -> pd.DataFrame:
        """Get the loaded and cleaned dataset"""
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        return self.df
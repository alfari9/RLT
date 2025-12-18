"""
Data Validation Pipeline
Validates data quality and schema before processing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates input data for quality and schema compliance"""
    
    def __init__(self, schema: Dict):
        self.schema = schema
        self.validation_results = {}
    
    def validate_missing_values(self, df: pd.DataFrame, threshold: float = 0.3) -> bool:
        """Check if missing values exceed threshold"""
        missing_ratio = df.isnull().sum() / len(df)
        failed_columns = missing_ratio[missing_ratio > threshold]
        
        if len(failed_columns) > 0:
            logger.warning(f"Columns exceeding missing value threshold: {failed_columns.to_dict()}")
            self.validation_results['missing_values'] = False
            return False
        
        self.validation_results['missing_values'] = True
        return True
    
    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """Validate data types match expected schema"""
        for column, expected_type in self.schema.get('dtypes', {}).items():
            if column in df.columns:
                if df[column].dtype != expected_type:
                    logger.warning(f"Column {column} has type {df[column].dtype}, expected {expected_type}")
                    self.validation_results['data_types'] = False
                    return False
        
        self.validation_results['data_types'] = True
        return True
    
    def validate_value_ranges(self, df: pd.DataFrame) -> bool:
        """Validate numeric columns are within expected ranges"""
        for column, ranges in self.schema.get('ranges', {}).items():
            if column in df.columns:
                min_val, max_val = ranges
                if df[column].min() < min_val or df[column].max() > max_val:
                    logger.warning(f"Column {column} values outside expected range [{min_val}, {max_val}]")
                    self.validation_results['value_ranges'] = False
                    return False
        
        self.validation_results['value_ranges'] = True
        return True
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Run all validation checks"""
        logger.info("Starting data validation...")
        
        checks = [
            self.validate_missing_values(df),
            self.validate_data_types(df),
            self.validate_value_ranges(df)
        ]
        
        all_passed = all(checks)
        logger.info(f"Validation complete. Status: {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed, self.validation_results


class DataPreprocessor:
    """Preprocesses data for ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values using specified strategy"""
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        if strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
        
        logger.info(f"Missing values handled using {strategy} strategy")
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method"""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        logger.info(f"Outliers removed. Rows remaining: {len(df_clean)}")
        return df_clean
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        df_processed = self.handle_missing_values(df)
        df_processed = self.remove_outliers(df_processed)
        
        logger.info("Preprocessing complete")
        return df_processed


if __name__ == "__main__":
    # Example usage
    schema = {
        'dtypes': {},
        'ranges': {}
    }
    
    validator = DataValidator(schema)
    preprocessor = DataPreprocessor()
    
    logger.info("Data validation and preprocessing pipeline initialized")

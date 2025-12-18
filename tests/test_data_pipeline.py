"""
Tests for data pipeline
"""
import pytest
import pandas as pd
import numpy as np
from src.pipelines.data_pipeline import DataValidator, DataPreprocessor


def test_data_validator_initialization():
    """Test DataValidator initialization"""
    schema = {'dtypes': {}, 'ranges': {}}
    validator = DataValidator(schema)
    assert validator.schema == schema
    assert validator.validation_results == {}


def test_data_preprocessor_initialization():
    """Test DataPreprocessor initialization"""
    preprocessor = DataPreprocessor()
    assert preprocessor.scalers == {}
    assert preprocessor.encoders == {}


def test_handle_missing_values():
    """Test missing value handling"""
    # Create sample data with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': ['a', 'b', 'c', 'd']
    })
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df, strategy='mean')
    
    # Check no missing values remain in numeric columns
    assert df_clean['A'].isnull().sum() == 0
    assert df_clean['B'].isnull().sum() == 0


def test_validate_missing_values():
    """Test missing value validation"""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9]
    })
    
    schema = {'dtypes': {}, 'ranges': {}}
    validator = DataValidator(schema)
    
    result = validator.validate_missing_values(df, threshold=0.3)
    assert result == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

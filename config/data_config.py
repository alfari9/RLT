"""
Data Pipeline Configuration
"""

# Data validation schema
DATA_SCHEMA = {
    'iris': {
        'dtypes': {
            'sepal_length': 'float64',
            'sepal_width': 'float64',
            'petal_length': 'float64',
            'petal_width': 'float64'
        },
        'ranges': {
            'sepal_length': (0, 10),
            'sepal_width': (0, 10),
            'petal_length': (0, 10),
            'petal_width': (0, 10)
        }
    }
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'missing_value_strategy': 'mean',
    'outlier_method': 'iqr',
    'outlier_threshold': 1.5,
    'scaling_method': 'standard'
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'max_missing_ratio': 0.3,
    'min_samples': 50,
    'max_duplicate_ratio': 0.1
}

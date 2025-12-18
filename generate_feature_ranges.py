import pandas as pd
import json
import numpy as np

# Load datasets_metadata.json to get exact feature names
with open('datasets_metadata.json', 'r') as f:
    datasets_metadata = json.load(f)

# Configuration complète pour chaque dataset
datasets_config = {
    'breast_cancer': {
        'file': 'uci-datasets/breast_cancer_wisconsin__diagnostic_.csv',
        'has_header': False,
        'col_names': ['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 
                      'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points',
                      'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error',
                      'perimeter error', 'area error', 'smoothness error', 'compactness error',
                      'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
                      'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
                      'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
                      'worst fractal dimension'],
        'target': 'diagnosis',
        'exclude': ['id', 'diagnosis']
    },
    'boston_housing': {
        'file': 'uci-datasets/boston_housing.csv',
        'has_header': False,
        'col_names': ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv'],
        'target': 'medv',
        'exclude': ['medv']
    },
    'parkinsons': {
        'file': 'uci-datasets/parkinsons.csv',
        'has_header': True,
        'target': 'status',
        'exclude': ['name', 'status']
    },
    'sonar': {
        'file': 'uci-datasets/sonar.csv',
        'has_header': False,
        'target': 'target',
        'exclude': ['target'],
        'use_metadata_names': True  # Use feature names from datasets_metadata.json
    },
    'white_wine_quality': {
        'file': 'uci-datasets/white_wine_quality.csv',
        'has_header': True,
        'sep': ';',
        'target': 'quality',
        'exclude': ['quality'],
        'rename_to_underscore': True  # Convert "fixed acidity" -> "fixed_acidity"
    },
    'red_wine_quality': {
        'file': 'uci-datasets/red_wine_quality.csv',
        'has_header': True,
        'sep': ';',
        'target': 'quality',
        'exclude': ['quality'],
        'rename_to_underscore': True
    },
    'parkinson_oxford': {
        'file': 'uci-datasets/parkinson_oxford.csv',
        'has_header': True,
        'target': 'total_UPDRS',
        'exclude': ['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS']
    },
    'ozone': {
        'file': 'uci-datasets/ozone.csv',
        'has_header': False,
        'target': 'ozone_reading',
        'exclude': ['date', 'ozone_reading'],
        'use_metadata_names': True  # Use feature names from datasets_metadata.json
    },
    'concrete_strength': {
        'file': 'uci-datasets/concrete_compressive_strength.csv',
        'has_header': True,
        'target': 'strength',
        'exclude': ['strength']
    },
    'auto_mpg': {
        'file': 'uci-datasets/auto_mpg.csv',
        'has_header': False,
        'col_names': ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin'],
        'target': 'mpg',
        'exclude': ['mpg'],
        'one_hot_origin': True  # Handle origin_2, origin_3 from origin column
    }
}

feature_ranges = {}

for dataset_name, config in datasets_config.items():
    print(f'Processing {dataset_name}...')
    
    sep = config.get('sep', ',')
    if config.get('has_header', True):
        df = pd.read_csv(config['file'], sep=sep)
    else:
        col_names = config.get('col_names', None)
        df = pd.read_csv(config['file'], sep=sep, header=None, names=col_names)
    
    # Get expected feature names from datasets_metadata.json
    expected_features = [f.lower() for f in datasets_metadata[dataset_name]['feature_names']]
    
    exclude = config.get('exclude', [])
    feature_cols = [c for c in df.columns if c not in exclude]
    df_features = df[feature_cols].select_dtypes(include=[np.number])
    
    feature_ranges[dataset_name] = {}
    
    # Special case: use feature names from metadata (sonar, ozone)
    if config.get('use_metadata_names', False):
        numeric_cols = df_features.columns.tolist()
        for i, meta_name in enumerate(expected_features):
            if i < len(numeric_cols):
                col_data = df_features[numeric_cols[i]].dropna()
                feature_ranges[dataset_name][meta_name] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'distribution': 'normal'
                }
    
    # Special case: rename spaces to underscores (wine datasets)
    elif config.get('rename_to_underscore', False):
        for col in df_features.columns:
            col_data = df_features[col].dropna()
            col_key = col.lower().replace(' ', '_')
            feature_ranges[dataset_name][col_key] = {
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'distribution': 'normal'
            }
    
    # Special case: auto_mpg with one-hot encoded origin
    elif config.get('one_hot_origin', False):
        for col in df_features.columns:
            col_data = df_features[col].dropna()
            col_key = str(col).lower()
            
            if col_key == 'origin':
                # One-hot encoding: origin_2 and origin_3 (origin_1 is the reference)
                feature_ranges[dataset_name]['origin_2'] = {
                    'min': 0.0,
                    'max': 1.0,
                    'mean': float((df_features['origin'] == 2).mean()),
                    'std': float((df_features['origin'] == 2).std()),
                    'distribution': 'normal'
                }
                feature_ranges[dataset_name]['origin_3'] = {
                    'min': 0.0,
                    'max': 1.0,
                    'mean': float((df_features['origin'] == 3).mean()),
                    'std': float((df_features['origin'] == 3).std()),
                    'distribution': 'normal'
                }
            else:
                feature_ranges[dataset_name][col_key] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'distribution': 'normal'
                }
    
    # Default case
    else:
        for col in df_features.columns:
            col_data = df_features[col].dropna()
            col_key = str(col).lower()
            feature_ranges[dataset_name][col_key] = {
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'distribution': 'normal'
            }
    
    print(f'  -> {len(feature_ranges[dataset_name])} features')

# Sauvegarder
with open('feature_ranges.json', 'w') as f:
    json.dump(feature_ranges, f, indent=2)

print('\n✅ feature_ranges.json saved!')

# Afficher quelques exemples
print('\n=== SAMPLE VALUES ===')
print('\nBreast Cancer (mean radius):')
print(f"  min: {feature_ranges['breast_cancer']['mean radius']['min']:.2f}")
print(f"  max: {feature_ranges['breast_cancer']['mean radius']['max']:.2f}")

print('\nBoston Housing (crim):')
print(f"  min: {feature_ranges['boston_housing']['crim']['min']:.4f}")
print(f"  max: {feature_ranges['boston_housing']['crim']['max']:.2f}")

print('\nConcrete Strength (cement):')
print(f"  min: {feature_ranges['concrete_strength']['cement']['min']:.1f}")
print(f"  max: {feature_ranges['concrete_strength']['cement']['max']:.1f}")

"""
Flask API for Model Serving and Monitoring
RESTful API endpoints for model predictions and health checks
"""
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
model_metadata = {}


def load_model(model_path: str = 'models/best_model.pkl'):
    """Load trained model"""
    global model
    
    if Path(model_path).exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return True
    else:
        logger.warning(f"Model not found at {model_path}")
        return False


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'ML Model API',
        'version': '1.0.0',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(health_status), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0].tolist()
        else:
            probabilities = None
        
        response = {
            'prediction': int(prediction[0]),
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {prediction[0]}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        response = {
            'predictions': predictions.tolist(),
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction made: {len(predictions)} samples")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/metrics')
def metrics():
    """Model metrics endpoint"""
    metrics_file = Path('metrics/metrics_history.json')
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data), 200
    else:
        return jsonify({'message': 'No metrics available'}), 404


@app.route('/model_info')
def model_info():
    """Model information endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_type': type(model).__name__,
        'features': getattr(model, 'n_features_in_', 'N/A'),
        'classes': getattr(model, 'classes_', 'N/A').tolist() if hasattr(model, 'classes_') else 'N/A'
    }
    
    return jsonify(info), 200


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False)

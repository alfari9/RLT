# Deployment Guide

## 🚀 Deployment Options

### Option 1: Local Deployment

#### Step 1: Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Prepare Models
```bash
# Train and save models
python -c "from src.pipelines.train_pipeline import ModelTrainer; print('Training models...')"
```

#### Step 3: Start Services
```bash
# Terminal 1: Start API
python dashboard/api.py

# Terminal 2: Start Dashboard
streamlit run dashboard/app.py

# Terminal 3: Start MLflow
mlflow ui --backend-store-uri ./mlruns --port 5001
```

### Option 2: Docker Deployment

#### Step 1: Build Images
```bash
# Build all services
docker-compose build
```

#### Step 2: Start Services
```bash
# Start all services in detached mode
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f ml-api
```

#### Step 3: Verify Deployment
```bash
# Test API health
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 5.1, "feature2": 3.5}'
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

#### AWS Deployment with EC2

1. **Launch EC2 Instance**
   - Instance type: t2.medium or higher
   - OS: Ubuntu 20.04 LTS
   - Security group: Open ports 5000, 8501, 5001

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip docker.io docker-compose -y
   ```

3. **Deploy Application**
   ```bash
   git clone https://github.com/alfari9/RLT.git
   cd RLT-Maha-Aloui
   docker-compose up -d
   ```

#### GCP Deployment with Cloud Run

1. **Build Container**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/ml-api
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy ml-api \
     --image gcr.io/PROJECT_ID/ml-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

#### Azure Deployment with Container Instances

1. **Create Resource Group**
   ```bash
   az group create --name ml-ops-rg --location eastus
   ```

2. **Deploy Container**
   ```bash
   az container create \
     --resource-group ml-ops-rg \
     --name ml-api \
     --image ml-model-api:latest \
     --ports 5000 \
     --cpu 2 --memory 4
   ```

## 🔒 Security Best Practices

### 1. API Security
```python
# Add authentication to Flask API
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    # ... prediction logic
```

### 2. Environment Variables
Create `.env` file:
```bash
FLASK_ENV=production
MODEL_PATH=/app/models/best_model.pkl
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
```

### 3. HTTPS Configuration
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Run Flask with SSL
python dashboard/api.py --cert cert.pem --key key.pem
```

## 📊 Monitoring in Production

### 1. Setup Prometheus Monitoring
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['ml-api:5000']
    metrics_path: '/metrics'
```

### 2. Setup Logging
```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('logs/app.log', maxBytes=10000000, backupCount=5)
logger.addHandler(handler)
```

### 3. Health Checks
```bash
# Add to cron for automated health checks
*/5 * * * * curl http://localhost:5000/health || echo "API Down" | mail -s "Alert" admin@example.com
```

## 🔄 Update and Rollback

### Update Models
```bash
# Train new model
python src/pipelines/train_pipeline.py

# Backup current model
cp models/best_model.pkl models/best_model_backup.pkl

# Deploy new model
docker-compose restart ml-api
```

### Rollback Strategy
```bash
# Rollback to previous version
git checkout HEAD~1
docker-compose down
docker-compose up -d

# Or restore model backup
cp models/best_model_backup.pkl models/best_model.pkl
docker-compose restart ml-api
```

## 📈 Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  ml-api:
    deploy:
      replicas: 3
    # ... other configurations
```

### Load Balancer (Nginx)
```nginx
upstream ml_api {
    server ml-api-1:5000;
    server ml-api-2:5000;
    server ml-api-3:5000;
}

server {
    listen 80;
    location / {
        proxy_pass http://ml_api;
    }
}
```

## 🧪 Testing Deployment

### Automated Testing
```bash
# Run integration tests
pytest tests/integration/ -v

# Load testing with locust
locust -f tests/load_test.py --host=http://localhost:5000
```

### Performance Benchmarking
```bash
# Apache Bench
ab -n 1000 -c 10 http://localhost:5000/predict

# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/predict
```

## 📱 Maintenance

### Daily Tasks
- Check service health
- Monitor resource usage
- Review error logs

### Weekly Tasks
- Update dependencies
- Backup models and data
- Review performance metrics

### Monthly Tasks
- Retrain models with new data
- Security updates
- Capacity planning review

## 🆘 Troubleshooting

### Common Issues

**Issue: Container won't start**
```bash
# Check logs
docker-compose logs ml-api

# Rebuild container
docker-compose build --no-cache ml-api
docker-compose up -d ml-api
```

**Issue: Model not loading**
```bash
# Verify model file exists
ls -lh models/best_model.pkl

# Check permissions
chmod 644 models/best_model.pkl
```

**Issue: High memory usage**
```bash
# Monitor resources
docker stats

# Increase memory limit in docker-compose.yml
services:
  ml-api:
    mem_limit: 4g
```

## 📞 Support

For deployment issues:
1. Check logs: `docker-compose logs`
2. Review documentation
3. Open GitHub issue
4. Contact: maha.aloui@example.com

---

**Deployment Date:** 2025-12-18  
**Version:** 1.0.0  
**Maintainer:** Maha Aloui

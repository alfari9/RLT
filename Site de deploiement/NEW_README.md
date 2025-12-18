# 🌲 RLT Model Comparison Platform - Deployment Site

Modern multi-page web application for comparing RLT (Reinforcement Learning Trees) with state-of-the-art machine learning models across 10 UCI datasets.

## ✨ What's New

### 🏠 Multi-Page Architecture
- **Home Page** (`/`) - Beautiful landing page with platform overview
- **Datasets Page** (`/datasets`) - Interactive grid showing all 10 datasets
- **Prediction Page** (`/predict/<dataset>`) - Make predictions on selected dataset
- **Global Insights** (`/insights`) - Comprehensive analytics dashboard with charts

### 🎨 Modern Design System
- **Fresh Color Palette**: Deep purple (#5B21B6), emerald green (#059669), vibrant cyan (#06B6D4)
- **Responsive Layout**: Perfect on desktop, tablet, and mobile devices
- **Modern Components**: Gradient buttons, glassmorphism cards, smooth animations
- **Professional Typography**: Clean, readable fonts with proper hierarchy

### 📊 Global Insights Dashboard
- **Performance Charts**: Compare all 5 models across 10 datasets
- **Complexity Scatter Plot**: Visualize dataset dimensionality
- **Performance Matrix**: Detailed table with all metrics
- **Win Rate Analysis**: See which model performs best overall

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask numpy pandas scikit-learn
```

### 2. Run the Server
```bash
python app.py
```

### 3. Open in Browser
Navigate to: **http://localhost:5000**

## 📁 File Structure

```
Site de deploiement/
├── app.py                      # Flask backend with 4 routes
├── templates/
│   ├── home.html              # Landing page
│   ├── datasets.html          # Dataset selection
│   ├── predict.html           # Prediction interface
│   └── insights.html          # Analytics dashboard
├── static/
│   ├── css/
│   │   ├── style.css         # Base styles
│   │   └── modern-styles.css # New modern components
│   └── js/
│       ├── main.js           # Prediction page logic
│       └── insights.js       # Charts & analytics
├── models/                    # Trained ML models (10 datasets)
└── requirements.txt
```

## 🎯 Features

### Dataset Selection
- **10 UCI Datasets**: Breast Cancer, Boston Housing, Sonar, Wine Quality, Parkinson, Concrete, Auto MPG, Ozone
- **Visual Cards**: Each dataset shows type, samples, features
- **Quick Navigation**: Click to select and start predicting

### Real-time Predictions
- **5 ML Models**: RLT, Random Forest, Gradient Boosting, ExtraTrees, Lasso/Ridge
- **Smart Input**: Random fill feature with realistic values
- **Instant Results**: All models predict simultaneously
- **Performance Metrics**: See historical accuracy/R² scores

### Global Analytics
- **Interactive Charts**: Powered by Chart.js
- **Model Comparison**: Bar charts comparing performance
- **Win Rate**: Doughnut chart showing best-performing models
- **Complexity Analysis**: Scatter plot of features vs samples

## 🎨 Color Palette

```css
Primary Purple:  #5B21B6  /* Main brand color */
Emerald Green:   #059669  /* Secondary actions */
Vibrant Cyan:    #06B6D4  /* Accents */
Success Green:   #10B981  /* Positive feedback */
Danger Red:      #EF4444  /* Errors */
Warning Orange:  #F59E0B  /* Warnings */
```

## 📊 Supported Models

| Model | Key Features |
|-------|--------------|
| **RLT Forest** | Variable muting, linear splits, bandit methods |
| **Random Forest** | Bootstrap aggregating, feature randomness |
| **Gradient Boosting** | Sequential trees, error minimization |
| **ExtraTrees** | Extremely randomized splits |
| **Lasso/Ridge** | Linear regression with regularization |

## 🔧 API Endpoints

- `GET /` - Home page
- `GET /datasets` - Datasets selection page
- `GET /predict/<dataset>` - Prediction page for specific dataset
- `GET /insights` - Global analytics dashboard
- `GET /api/datasets` - JSON list of all datasets
- `GET /api/dataset/<name>/info` - JSON dataset metadata
- `POST /api/predict` - Make prediction with input features

## 📱 Responsive Design

The site is fully responsive and works perfectly on:
- 💻 Desktop (1920px+)
- 💻 Laptop (1366px - 1919px)
- 📱 Tablet (768px - 1365px)
- 📱 Mobile (320px - 767px)

## 🎓 Academic Reference

Based on research by:
**Zhu, R., Zeng, D., & Kosorok, M. R. (2015)**
*Reinforcement Learning Trees*
Journal of the American Statistical Association

## 📝 License

University Research Project - 2024

## 🤝 Contributing

This is a research project. For questions or improvements, please contact the project team.

---

**Built with ❤️ for ML research**

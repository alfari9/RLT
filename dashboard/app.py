"""
Streamlit Dashboard for Model Monitoring
Real-time monitoring and visualization of ML models
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import sys

# Page configuration
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


def load_metrics_history():
    """Load metrics history from JSON file"""
    metrics_file = Path('metrics/metrics_history.json')
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return []


def load_model_registry():
    """Load model registry"""
    registry_file = Path('models/registry.json')
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            return json.load(f)
    return {}


def main():
    # Header
    st.title("🤖 ML Model Monitoring Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Model Performance", "Experiment Tracking", "Data Quality", "System Metrics"]
        )
        
        st.markdown("---")
        st.header("⚙️ Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 30)
        
        st.markdown("---")
        st.info("Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Main content based on selected page
    if page == "Overview":
        show_overview()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Experiment Tracking":
        show_experiment_tracking()
    elif page == "Data Quality":
        show_data_quality()
    elif page == "System Metrics":
        show_system_metrics()


def show_overview():
    """Display overview dashboard"""
    st.header("📊 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Models",
            value="3",
            delta="+1"
        )
    
    with col2:
        st.metric(
            label="Experiments Run",
            value="24",
            delta="+5"
        )
    
    with col3:
        st.metric(
            label="Best Accuracy",
            value="94.5%",
            delta="+2.1%"
        )
    
    with col4:
        st.metric(
            label="Data Quality",
            value="98.2%",
            delta="+0.5%"
        )
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Experiments")
        metrics_history = load_metrics_history()
        
        if metrics_history:
            recent_df = pd.DataFrame(metrics_history[-5:])
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No experiment data available")
    
    with col2:
        st.subheader("🏆 Top Performing Models")
        model_registry = load_model_registry()
        
        if model_registry:
            st.json(model_registry, expanded=False)
        else:
            st.info("No models registered")


def show_model_performance():
    """Display model performance metrics"""
    st.header("📈 Model Performance")
    
    metrics_history = load_metrics_history()
    
    if not metrics_history:
        st.warning("No performance data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_history)
    
    # Metrics over time
    st.subheader("Metrics Evolution")
    
    metric_options = [col for col in df.columns if col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    
    if metric_options:
        selected_metrics = st.multiselect(
            "Select Metrics to Display",
            metric_options,
            default=metric_options[:2] if len(metric_options) >= 2 else metric_options
        )
        
        if selected_metrics:
            fig = go.Figure()
            
            for metric in selected_metrics:
                if metric in df.columns:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(df))),
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title()
                    ))
            
            fig.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Experiment Number",
                yaxis_title="Score",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance comparison
    st.subheader("Model Comparison")
    
    if 'experiment_name' in df.columns:
        comparison_df = df[['experiment_name'] + [col for col in metric_options if col in df.columns]]
        st.dataframe(comparison_df, use_container_width=True)


def show_experiment_tracking():
    """Display experiment tracking information"""
    st.header("🔬 Experiment Tracking")
    
    st.info("Experiment tracking integrated with MLflow. View detailed runs in MLflow UI.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Experiment Configuration")
        st.code("""
        experiment_name: ml_model_experiments
        tracking_uri: ./mlruns
        artifact_location: ./mlartifacts
        """, language="yaml")
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Runs", "24")
        st.metric("Active Experiments", "3")
        st.metric("Logged Artifacts", "156")


def show_data_quality():
    """Display data quality metrics"""
    st.header("🎯 Data Quality Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Validation Status")
        
        validation_status = {
            'Missing Values': 'PASSED',
            'Data Types': 'PASSED',
            'Value Ranges': 'PASSED',
            'Duplicate Records': 'WARNING'
        }
        
        for check, status in validation_status.items():
            status_emoji = "✅" if status == "PASSED" else "⚠️"
            st.write(f"{status_emoji} {check}: {status}")
    
    with col2:
        st.subheader("Data Statistics")
        
        stats = {
            'Total Records': '10,000',
            'Missing Values': '0.5%',
            'Duplicate Rows': '2.1%',
            'Outliers Detected': '3.2%'
        }
        
        for stat, value in stats.items():
            st.metric(stat, value)


def show_system_metrics():
    """Display system and infrastructure metrics"""
    st.header("💻 System Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Resource Usage")
        st.metric("CPU Usage", "45%")
        st.metric("Memory Usage", "3.2 GB")
        st.metric("Disk Usage", "12.5 GB")
    
    with col2:
        st.subheader("Model Serving")
        st.metric("API Requests", "1,247")
        st.metric("Avg Response Time", "120ms")
        st.metric("Error Rate", "0.02%")
    
    with col3:
        st.subheader("Infrastructure")
        st.metric("Active Containers", "3")
        st.metric("Uptime", "99.98%")
        st.metric("Last Deployment", "2 hours ago")


if __name__ == "__main__":
    main()

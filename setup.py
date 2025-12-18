"""
Setup configuration for MLOps project
"""
from setuptools import setup, find_packages

setup(
    name="mlops-rlt",
    version="1.0.0",
    author="Maha Aloui",
    description="MLOps Machine Learning Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.3.2",
        "pandas>=2.1.4",
        "numpy>=1.26.2",
    ],
)

# car-price-prediction
Car Price Prediction - A comprehensive machine learning project that predicts used car prices using regression algorithms. Developed during my Data Science internship at Acmegrade (DS Aug 2025 batch), this project demonstrates end-to-end ML pipeline development from data exploration to deployment.

## 🎯 Project Highlights:
- **ML Algorithms**: Linear Regression, Random Forest, Extra Trees
- **Performance**: 94.85% R² score with £1,156 average error
- **Features**: Interactive dashboard, feature importance analysis
- **Tech Stack**: Python, Scikit-learn, Pandas, Matplotlib, Jupyter

## 📊 Key Features:
- Complete EDA and data preprocessing pipeline
- Multiple model comparison and evaluation
- Feature importance analysis and business insights
- Interactive prediction dashboard
- Model deployment with pickle files

## 🏆 Acmegrade Internship:
This project was developed as part of my Data Science internship at Acmegrade (August 2025 batch), showcasing practical implementation of machine learning concepts in real-world scenarios.

---------------

# 🚗 Car Price Prediction Project

<div align="center">

**Made by Debadatta Rout**  
*Internship at Acmegrade in Data Science*  
*Batch - DS Aug 2025*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

*A Comprehensive Machine Learning Project for Predicting Audi Car Prices*

</div>

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📊 Dataset Description](#-dataset-description)
- [🛠️ Project Architecture](#️-project-architecture)
- [📈 Project Workflow](#-project-workflow)
- [🔍 Phase 1: Data Collection & Exploration](#-phase-1-data-collection--exploration)
- [🔄 Phase 2: Data Preprocessing](#-phase-2-data-preprocessing)
- [🤖 Phase 3: Model Development](#-phase-3-model-development)
- [📊 Phase 4: Model Evaluation](#-phase-4-model-evaluation)
- [🔎 Phase 5: Feature Analysis](#-phase-5-feature-analysis)
- [💾 Phase 6: Model Deployment](#-phase-6-model-deployment)
- [🎛️ Phase 7: Interactive Dashboard](#️-phase-7-interactive-dashboard)
- [📈 Results & Performance](#-results--performance)
- [🎯 Key Findings](#-key-findings)
- [🚀 How to Use](#-how-to-use)
- [🔮 Future Enhancements](#-future-enhancements)
- [📚 Conclusion](#-conclusion)

## 🎯 Project Overview

This project represents my comprehensive journey through building a complete machine learning pipeline for predicting Audi car prices. As part of my Data Science internship at Acmegrade, I developed an end-to-end solution that transforms raw car data into accurate price predictions using advanced regression techniques.

The project demonstrates my understanding of:
- **Data preprocessing** and feature engineering
- **Multiple machine learning algorithms**
- **Model evaluation** and selection
- **Feature importance analysis**
- **Model deployment** and interactive dashboard creation

**Business Problem**: Develop a reliable model to predict used Audi car prices based on various features like model, year, mileage, transmission, and other specifications.

## 📊 Dataset Description

The project utilizes the `audi.csv` dataset containing comprehensive information about Audi cars with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `model` | Car model (A1, A3, A4, A6, etc.) | Categorical |
| `year` | Manufacturing year | Numerical |
| `price` | Target variable - car price in £ | Numerical |
| `transmission` | Transmission type | Categorical |
| `mileage` | Distance traveled in miles | Numerical |
| `fuelType` | Type of fuel | Categorical |
| `tax` | Annual road tax in £ | Numerical |
| `mpg` | Miles per gallon | Numerical |
| `engineSize` | Engine capacity in liters | Numerical |

**Dataset Statistics**:
- Total Records: 10,000+ cars
- Features: 9 attributes
- Time Period: Various manufacturing years
- Price Range: £5,000 - £80,000

## 🛠️ Project Architecture

*[Insert Project Architecture Diagram Here]*

The project follows a structured 7-phase approach:

1. **Data Collection & Exploration**
2. **Data Preprocessing**
3. **Model Development**
4. **Model Evaluation**
5. **Feature Analysis**
6. **Model Deployment**
7. **Interactive Dashboard**

## 📈 Project Workflow

*[Insert Detailed Workflow Diagram Here]*

## 🔍 Phase 1: Data Collection & Exploration

### Step 1.1: Environment Setup and Library Imports
```python
# Essential libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn packages for machine learning

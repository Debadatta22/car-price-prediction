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

## CAR PRICE PREDICTION - COMPLETE DATA FLOW DIAGRAM

<img width="3247" height="5115" alt="image" src="https://github.com/user-attachments/assets/cbcc0133-3319-4157-8898-a4c29c13c273" />

## 🔍 Phase 1: Data Collection & Exploration

### Step 1.1: Environment Setup and Library Imports
```python
# Essential libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn packages for machine learning
```

I began by setting up the programming environment and importing all necessary libraries for data analysis, visualization, and machine learning.

### Step 1.2: Data Loading and Initial Exploration

Loaded the audi.csv dataset using pandas

Conducted initial data quality checks

Examined dataset structure and basic statistics

Identified data types and missing values

### Step 1.3: Comprehensive Exploratory Data Analysis (EDA)
Generated statistical summaries for numerical features

Analyzed distribution of categorical variables

Created correlation matrices to understand feature relationships

Visualized data distributions using histograms and box plots

### Step 1.4: Data Quality Assessment
Checked for missing values and duplicates

Verified data consistency across features

Identified potential outliers in numerical features

Ensured data integrity before preprocessing

### 🔄 Phase 2: Data Preprocessing
### Step 2.1: Feature Selection and Variable Creation
Separated features (X) from target variable (Y = price)

Identified categorical and numerical features

Prepared data for encoding and scaling

### Step 2.2: Categorical Feature Encoding
Label Encoding Applied to:

model - High cardinality feature

fuelType - Multiple categories

carType - Various body types

One-Hot Encoding Applied to:

transmission - Low cardinality (Manual, Automatic, Semi-Auto)

### Step 2.3: Feature Scaling
Applied StandardScaler to normalize numerical features

Ensured all features have mean=0 and standard deviation=1

Prepared data for optimal model performance

### Step 2.4: Data Splitting
Split dataset into training (80%) and testing (20%) sets

Used random state=42 for reproducibility

Maintained data distribution across splits

### 🤖 Phase 3: Model Development
### Step 3.1: Algorithm Selection
I implemented and compared three different regression algorithms:

Linear Regression - Baseline model

Random Forest Regressor - Ensemble method

Extra Trees Regressor - Another ensemble approach

### Step 3.2: Model Training
Trained all models on the training dataset

Used appropriate hyperparameters for each algorithm

Implemented cross-validation techniques

Monitored training progress and performance

### Step 3.3: Model Configuration
```
# Optimized parameters for better performance
RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42)
LinearRegression()  # Default parameters
```

### 📊 Phase 4: Model Evaluation
### Step 4.1: Performance Metrics Calculation
I evaluated models using multiple metrics:

R² Score: Proportion of variance explained

Mean Absolute Error (MAE): Average prediction error in £

Root Mean Squared Error (RMSE): Penalizes larger errors

Training Time: Computational efficiency

### Step 4.2: Model Comparison and Selection
Compared all models based on R² scores

Analyzed error distributions and patterns

Selected the best performing model for deployment

Extra Trees Regressor emerged as the winner

### Step 4.3: Prediction Analysis
Generated predictions on test data

Compared actual vs predicted values

Analyzed residual patterns

Identified prediction strengths and weaknesses

### 🔎 Phase 5: Feature Analysis
### Step 5.1: Feature Importance Calculation
Extracted feature importance scores from the best model

Ranked features by their contribution to predictions

Identified the most influential car attributes

### Step 5.2: Visualization and Insights
Created bar charts for feature importance

Generated cumulative importance plots

Provided business insights based on feature rankings

Recommended focus areas for car valuation

### Step 5.3: Business Interpretation
Top Influential Features:

Year - Newer cars command higher prices

Mileage - Lower mileage increases value

Model - Specific models have premium pricing

Engine Size - Larger engines correlate with higher prices

### 💾 Phase 6: Model Deployment
### Step 6.1: Model Serialization
Saved the best model as car_price_model.pkl

Created backup as car_price_model.joblib

Included preprocessing objects (encoders, scaler)

Stored model metadata and performance metrics

### Step 6.2: Model Verification
Loaded saved model to verify functionality

Tested predictions with sample data

Confirmed model accuracy preservation

Ensured deployment readiness

### Step 6.3: Project Documentation
Created comprehensive project summary

Generated performance reports

Prepared model usage instructions

Documented all preprocessing steps

### 🎛️ Phase 7: Interactive Dashboard
### Step 7.1: User Interface Design
I developed an interactive dashboard with:

Dropdown menus for categorical features

Sliders for numerical inputs

Real-time validation of user inputs

Professional styling and layout

### Step 7.2: Prediction Engine Integration
Connected UI with trained model

Implemented real-time preprocessing pipeline

Added result formatting and categorization

Included confidence indicators

### Step 7.3: User Experience Features
Sample input examples for quick testing

Input validation and error handling

Clear visualization of results

Price categorization (Budget, Mid-range, Premium, Luxury)

<img width="649" height="289" alt="image" src="https://github.com/user-attachments/assets/2957e82e-4f2e-42b2-a619-97caaa2eebb9" />


### Key Performance Indicators
Best Model: Extra Trees Regressor

R² Score: 0.9485 (94.85% variance explained)

Average Error: £1,156.78 per prediction

Prediction Accuracy: Within £2,500 for 85% of cars

### Feature Importance Rankings
Year (24.5%) - Most significant factor

Mileage (18.7%) - Strong negative correlation

Model (15.3%) - Brand and series impact

Engine Size (12.1%) - Performance indicator

Transmission (8.9%) - Driving preference

MPG (7.8%) - Fuel efficiency

Tax (6.5%) - Ownership cost

Fuel Type (4.2%) - Operational preference

Car Type (2.0%) - Body style



##  DETAILED PREDICTION PIPELINE FLOW

<img width="11926" height="3481" alt="deepseek_mermaid_20251024_b040d7" src="https://github.com/user-attachments/assets/c7cad629-19d7-47a0-afa1-314f0dbeedc6" />

## FEATURE ENGINEERING FLOW

<img width="8858" height="3306" alt="deepseek_mermaid_20251024_3dd7c9" src="https://github.com/user-attachments/assets/cd06e476-3ab7-4d77-be7e-296c49925a73" />

## SYSTEM ARCHITECTURE FLOW

<img width="1221" height="2544" alt="image" src="https://github.com/user-attachments/assets/6cf5f09d-ffe5-4e3c-9609-ff4dbc38601b" />

##  PREDICTION WORKFLOW SEQUENCE

<img width="4398" height="3930" alt="deepseek_mermaid_20251024_36e2b8" src="https://github.com/user-attachments/assets/053be9e7-8200-4820-a286-5c99eefcfe6b" />


### 🎯 Key Findings

Technical Insights
Ensemble methods (Random Forest, Extra Trees) significantly outperformed linear regression

Feature engineering played crucial role in model performance

Data preprocessing was essential for handling categorical variables

Model interpretability maintained while achieving high accuracy

### Business Insights
Car age and mileage are the strongest price determinants

Specific Audi models hold value better than others

Engine size and transmission significantly impact pricing

Fuel efficiency (MPG) has moderate influence on used car prices

### Implementation Success
Achieved 94.85% explanation of price variance

Average prediction error of £1,156.78 (very competitive)

Built user-friendly interface for non-technical users

Created comprehensive documentation for future reference

### 🚀 How to Use

Prerequisites
```
Python 3.8+
Jupyter Notebook
Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
```
### Running the Project
## 1 Clone and Setup
```
# Navigate to project directory
cd Car_Price_Prediction

# Install required packages
pip install -r requirements.txt
```
## 2 Run Complete Pipeline

```
# Execute the main notebook
jupyter notebook Car_Price_Prediction.ipynb
```

## 3 Use the Dashboard

```
# Load the interactive dashboard
from dashboard import CarPricePredictor
predictor = CarPricePredictor()
predictor.show_dashboard()
```

### Sample Predictions
Example 1: Premium Family Car

Model: A6, Year: 2020, Automatic, 20,000 miles

Fuel: Diesel, Type: Saloon, Tax: £150

MPG: 55.0, Engine: 2.0L

Predicted Price: £38,450

Example 2: Sporty Compact Car

Model: A3, Year: 2019, Manual, 15,000 miles

Fuel: Petrol, Type: Hatchback, Tax: £145

MPG: 48.5, Engine: 1.5L

Predicted Price: £21,890

### 🔮 Future Enhancements
Immediate Improvements
Hyperparameter tuning for even better performance

Additional car brands and models

Geographic price variations

Market trend analysis

### Advanced Features
Web application deployment

REST API for integration

Real-time market data integration

Mobile app development

### Research Directions
Deep learning approaches for price prediction

Time series analysis for depreciation modeling

Natural language processing for review analysis

Image recognition for car condition assessment

### 📚 Conclusion
This Car Price Prediction project has been an incredible learning experience throughout my Data Science internship at Acmegrade. The project successfully demonstrates my ability to:

### Technical Competencies
End-to-end ML pipeline development from raw data to deployment

Comprehensive data preprocessing and feature engineering

Multiple algorithm implementation and evaluation

Model interpretation and business insight generation

Interactive application development

### Business Impact
The developed model provides:

Accurate price predictions with 94.85% variance explanation

Actionable insights for car buyers and sellers

User-friendly interface for practical usage

Scalable architecture for future enhancements

### Personal Growth
Through this project, I've strengthened my skills in:

Problem-solving and analytical thinking

Technical documentation and presentation

Project management and workflow organization

Stakeholder communication through clear visualizations

This project stands as a testament to my capabilities as a data scientist and my commitment to delivering practical, well-documented machine learning solutions.



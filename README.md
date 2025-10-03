# Crop Yield Prediction Pipeline

This repository contains a **stacking ensemble model** for predicting crop yields in India using historical data. The pipeline is designed to handle numerical, categorical, and skewed features and provides highly accurate predictions using state-of-the-art machine learning models.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Features](#features)  
- [Pipeline](#pipeline)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Evaluation](#evaluation)  
- [Example Prediction](#example-prediction)  
- [License](#license)  

---

## Project Overview

Accurately predicting crop yield is critical for farmers, policymakers, and agri-businesses. This project leverages machine learning to estimate yields based on factors such as area, production, rainfall, fertilizers, pesticides, crop type, season, and state.

---

## Dataset

The dataset used in this project can be downloaded from Kaggle:  

[Crop Yield Data - India](https://www.kaggle.com/datasets/saincoder404/crop-yield-data-india)  

The CSV should be saved as `crop_yield.csv` in the project root.

---

## Features

The model uses the following features:

- **Numerical (skewed)**: `Production` (log-transformed)  
- **Numerical (plain)**: `Area`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`, `Crop_Year`  
- **Categorical**: `Crop`, `Season`, `State`  

---

## Pipeline

1. **Preprocessing**

   - Log-transform skewed numerical features (`Production`)  
   - Standard scaling of numerical features  
   - One-hot encoding of categorical features (`Crop`, `Season`, `State`)  

2. **Stacking Ensemble**

   **Base learners (level-0):**  
   - LightGBM Regressor (`lgb`) – if installed  
   - XGBoost Regressor (`xgb`) – if installed  
   - HistGradientBoosting Regressor (`hgb`)  
   - RandomForest Regressor (`rf`)  

   **Meta-learner (level-1):**  
   - RandomForest Regressor (takes predictions from base learners + original features)  

3. **Target Transformation**

   - Log-transform applied to `Yield` during training  
   - Predictions are transformed back to original scale  

---

## Installation

```bash
git clone https://github.com/yourusername/crop-yield-prediction.git
cd crop-yield-prediction
pip install -r requirements.txt

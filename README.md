# House Price Prediction - A Machine Learning Case Study

## Problem Statement
The goal of this project is to predict the final sale prices of homes based on various features such as number of rooms, location, year built, etc. This dataset is part of a Kaggle competition **House Prices - Advanced Regression Techniques**.

We are trying to solve a **supervised regression** problem where the target variable is `SalePrice`.

## Dataset Overview
- **Source**: [Kaggle - House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Rows**: 1460
- **Columns**: 81 features
- **Types of Features**:
  - Numerical: `LotArea`, `YearBuilt`, `GrLivArea`
  - Categorical: `Neighborhood`, `HouseStyle`, `Exterior1st`

## Step 1: Initial Exploration
We started by loading the dataset and inspecting the first few rows:

code:
import pandas as pd
train_data = pd.read_csv('train.csv')
train_data.head()

## Step 2: Checking for Missing Values

We identified columns with missing values and summarized the amount:

code:
missing_values = train_data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
missing_values.head()

Top 5 columns with missing values:

PoolQC (1453 missing)

MiscFeature (1406 missing)

Alley (1369 missing)

Fence (1179 missing)

FireplaceQu (690 missing)

These features will be imputed or dropped depending on the significance.

## Step 3: Data Cleaning & Preprocessing

Next, we handle missing data. We will:

Fill missing values in categorical columns with 'None'

Drop columns with too many missing values (e.g., PoolQC, MiscFeature)


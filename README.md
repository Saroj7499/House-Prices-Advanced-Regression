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

After inspecting the dataset, we took the following actions:

**Dropped Columns with Too Many Missing Values**

We removed the following columns because they had over 1000 missing values, which means more than 70% of the data was missing:

code:
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')

This ensures that the model doesn't interpret missing values as unknown but rather as a valid category.

**Numerical Missing Values: Replaced with Median**
The column LotFrontage (linear feet of street connected to property) had 259 missing values. We filled it using the median value:

code:
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())

We chose the median to avoid skewing due to outliers.

**Final Check**
We verified that all missing values were successfully handled:
code:
train_data.isnull().sum().sum()  # Output: 0


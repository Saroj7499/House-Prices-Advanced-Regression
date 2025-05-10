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

## code:
import pandas as pd
train_data = pd.read_csv('train.csv')
train_data.head()

## Step 2: Checking for Missing Values

We identified columns with missing values and summarized the amount:

## code:
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

## code:
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')

This ensures that the model doesn't interpret missing values as unknown but rather as a valid category.

**Numerical Missing Values: Replaced with Median**
The column LotFrontage (linear feet of street connected to property) had 259 missing values. We filled it using the median value:

## code:
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())

We chose the median to avoid skewing due to outliers.

**Final Check**
We verified that all missing values were successfully handled:
code:
train_data.isnull().sum().sum()  # Output: 0

**Step 4: Handling Missing Values**

Dropping Columns with Too Many Missing Values
We observed that the columns PoolQC, MiscFeature, Alley, and Fence had a large number of missing values. These features likely represent rare or less informative characteristics, so we decided to drop them:

## code:
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
train_data.drop(columns=cols_to_drop, inplace=True)
drop(columns=...): removes specified columns.

inplace=True: makes the changes directly to train_data without needing reassignment.

 **Filling Missing Categorical Values**
We filled missing values in the FireplaceQu column with 'None', assuming that missing means no fireplace:

## code:
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')
Similarly, for the LotFrontage column (numerical), we filled missing values using the median of the column:

## code:
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
We chose median to avoid being affected by outliers.

This ensures the distribution remains balanced.

 ## Result
After cleaning, we verified that there were no more missing values:

## code:
train_data.isnull().sum().sum()
# Output: 0
This confirms that our dataset is now ready for further analysis and modeling.

**Step 5: Handling Missing Values (Continued)**

**Filling Missing Categorical Values with 'None'**

We continued to fill missing values in categorical columns with the placeholder value 'None'. This was done for columns where missing values likely indicate the absence of the feature (such as no garage, no basement, etc.).

Here are the columns for which we filled missing values with 'None':

## Columns: GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, MasVnrType, and Electrical.

## Code:

categorical_fill_none = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'Electrical']
for col in categorical_fill_none:
    train_data[col] = train_data[col].fillna('None')

## Explanation: 
The loop iterates through each column in the categorical_fill_none list and fills missing values with 'None', which is assumed to indicate the absence of that feature.

## Filling Missing Numerical Values with Median

Next, we handled missing values in numerical columns by filling them with the median value of the respective column. This method is used because the median is less sensitive to outliers compared to the mean, making it a good choice when handling missing data.

For the columns GarageYrBlt and MasVnrArea, we filled the missing values with their respective medians.

## Code:

train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].median())
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].median())

## Explanation:
The fillna() function is used to replace missing values in the specified columns with the median of that column. This ensures that the missing values are imputed in a way that maintains the overall distribution of the data.

## Result:
After executing these steps, we successfully filled the missing categorical and numerical values, ensuring that our dataset is now complete without any missing data.




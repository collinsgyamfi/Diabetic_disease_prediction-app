# Diabetic_disease_prediction-app
diabetic disease prediction application using a machine learning algorithm
# Diabetes Prediction Project

## Input Features for Testing the Project

- **Pregnancies**: Number of times the patient has been pregnant.
  - Example: `2`
  
- **Glucose**: Plasma glucose concentration at 2 hours in an oral glucose tolerance test.
  - Example: `120`
  
- **Blood Pressure**: Diastolic blood pressure (mm Hg).
  - Example: `70`
  
- **Skin Thickness**: Triceps skin fold thickness (mm).
  - Example: `20`
  
- **Insulin**: 2-Hour serum insulin (mu U/ml).
  - Example: `85`
  
- **BMI**: Body mass index (weight in kg/(height in m)^2).
  - Example: `28.5`
  
- **Diabetes Pedigree Function (DPF)**: A function scoring the likelihood of diabetes based on family history.
  - Example: `0.5`
  
- **Age**: Age of the patient (years).
  - Example: `30`

## Insights Summary of the diabetes.csv Dataset and Cleaning Methods

The `diabetes.csv` dataset contains patient health metrics and an Outcome column indicating diabetes diagnosis (1 = positive, 0 = negative). Here’s a summary of the dataset and cleaning recommendations:

### Dataset Overview
- **Columns**: 9 features (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age, Outcome).
- **Rows**: ~768 entries (estimated).
- **Target Variable**: Outcome (binary classification).

### Critical Observations
- **Missing Values**: Non-physiological zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI.
  - **Action**: Replace zeros with NaN and impute using median/mean or advanced methods (e.g., KNN).
  
- **Class Distribution**: Outcome=1 (diabetes) appears less frequent than Outcome=0.
  - **Action**: If needed, check for imbalance and apply techniques like SMOTE.
  
- **Outliers**: Extreme values (e.g., Insulin=846, BMI=67.1).
  - **Action**: Use IQR or Z-score to detect and cap/remove outliers.
  
- **Feature Ranges**: Variables like Insulin (0–846) and Age (21–81) have vastly different scales.
  - **Action**: Normalize/standardize features for modeling.
  
- **Data Entry Errors**: Validate extreme values against domain knowledge.

### Next Steps for Analysis
1. Clean data by handling missing values (replace 0s).
2. Check for outliers and decide how to handle them.
3. Explore correlations between features and the outcome.
4. Balance the dataset if there's significant class imbalance.
5. Scale features before building a predictive model.

## Detailed Explanation of Data Cleaning (Python Code)

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('diabetes.csv')

# Replace biological zeros with NaN
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, pd.NA)

# Impute missing values
imputer = SimpleImputer(strategy='median')
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = imputer.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

# Handle outliers using IQR
Q1 = df[['Insulin', 'BMI']].quantile(0.25)
Q3 = df[['Insulin', 'BMI']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['Insulin', 'BMI']] < (Q1 - 1.5 * IQR)) | (df[['Insulin', 'BMI']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Check class distribution
X = df.drop('Outcome', axis=1)
y = df['Outcome']
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Feature scaling
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Save cleaned data
cleaned_df = pd.DataFrame(X_res, columns=X.columns)
cleaned_df['Outcome'] = y_res
cleaned_df.to_csv('diabetes_processed.csv', index=False)

# Visualizations
sns.boxplot(data=cleaned_df)
plt.show()

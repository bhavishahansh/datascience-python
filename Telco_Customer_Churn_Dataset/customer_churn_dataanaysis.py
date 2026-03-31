import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df['MonthlyCharges'],df['tenure'], df['gender'])

df['TotalCharges'] = np.where(
    df['TotalCharges'] == ' ',
    np.nan,
    df['TotalCharges']
)

df['TotalCharges'] = df['TotalCharges'].astype(float)

median_tc = np.nanmedian(df['TotalCharges'])

df['TotalCharges'].fillna(median_tc,inplace=True)

df['TotalCharges'].isna().sum()

#df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

Q1 = np.percentile(df['MonthlyCharges'], 25)
Q3 = np.percentile(df['MonthlyCharges'], 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(lower)

df['MonthlyCharges'] = np.clip(
    df['MonthlyCharges'],
    lower,
    upper
)

df['avg_charge_per_month'] = (
    df['TotalCharges'] / np.maximum(df['tenure'], 1)
)

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
#plt.show()


sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
#plt.show()

sns.histplot(df['tenure'], bins=30, kde=True)
plt.title("Customer Tenure Distribution")
plt.show()


corr = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_charge_per_month']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#start from countplot graph
# start from here
'''

STEP 9: Exploratory Data Analysis (Seaborn Core)
9.1 Churn Distribution 
ask chat gpt why it is class imbalance?


✔ Customers with higher monthly charges churn more
✔ Low tenure customers are high-risk
✔ Long-term customers generate stable revenue
✔ Pricing strategy impacts churn significantly


Project: Customer Churn & Revenue Analysis

Analyzed 7K+ customer records using NumPy-based statistical techniques

Performed IQR-based outlier treatment and feature engineering

Conducted EDA using Seaborn to identify churn drivers

Delivered business insights to support retention strategies

'''
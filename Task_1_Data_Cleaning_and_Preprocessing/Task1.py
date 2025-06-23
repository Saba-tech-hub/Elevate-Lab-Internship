import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("C:\\Users\\Mohamed shapan\\Desktop\\SABA🙃\\Elevate Lab\\Task 1(Data Cleaning & Preprocessing)\\pokemon_data.csv")
print("Initial Dataset Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# 2. Handle missing values (mean for numerical, mode for categorical)

num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())


cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# 3. Encode categorical features using one-hot encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4. Normalize/standardize numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Visualize and remove outliers using boxplot and IQR
for col in num_cols:
    plt.figure(figsize=(6, 1))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

   
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]


print("\nCleaned Dataset Info:")
print(df.info())
print("\nSample Rows:\n", df.head())

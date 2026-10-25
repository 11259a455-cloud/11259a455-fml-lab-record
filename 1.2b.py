
import pandas as pd
import numpy as np

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Age": [23, np.nan, 22, 28, np.nan],
    "City": ["New York", "London", np.nan, "Paris", "Berlin"]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

print("\nNull values count in each column:")
print(df.isnull().sum())

df_clean = df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(df_clean)

df_filled = df.fillna(0)
print("\nDataFrame after filling missing values with 0:")
print(df_filled)

df_mean_filled = df.copy()
df_mean_filled["Age"] = df_mean_filled["Age"].fillna(df_mean_filled["Age"].mean())
print("\nDataFrame after filling 'Age' with mean value:")
print(df_mean_filled)

df_city_filled = df.copy()
df_city_filled["City"] = df_city_filled["City"].fillna("Unknown")
print("\nDataFrame after filling 'City' with 'Unknown':")
print(df_city_filled)
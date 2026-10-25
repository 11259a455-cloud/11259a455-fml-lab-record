import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data.csv")

print("Original Dataset:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

data.fillna(data.mean(numeric_only=True), inplace=True)

X = data[['Hours_Studied', 'Attendance', 'Assignments', 'Marks']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("\nNormalized Dataset:")
print(X_scaled.head())
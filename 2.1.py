import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")

X = data[['Hours_Studied', 'Attendance', 'Assignments']]
y = data['Result']

# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:")
print(X_train)
print("\nTesting Data:")
print(X_test)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
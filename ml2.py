-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("student-por.csv")

print("Original Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned Columns:")
print(df.columns.tolist())


print("\nMissing Values:")
print(df.isnull().sum())

# Fill numerical missing values
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset after cleaning:")
print(df.shape)

possible_targets = [
    "math_score",
    "maths_score",
    "mathematics_score",
    "math",
    "mathematics"
]

target = None

for column in possible_targets:
    if column in df.columns:
        target = column
        break

# If math score is not found, use the last numerical column
if target is None:

    numerical_columns = df.select_dtypes(include=np.number).columns

    if len(numerical_columns) > 0:
        target = numerical_columns[-1]
    else:
        print("\nNo numerical target column found.")
        exit()

print("\nTarget column selected:", target)
-

categorical_columns = df.select_dtypes(include="object").columns

encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

print("\nDataset after encoding:")
print(df.head())


X = df.drop(target, axis=1)
y = df[target]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(target)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ---------------------------------------------------------
# 8. CREATE LINEAR REGRESSION MODEL
# ---------------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

prin

y_pred = model.predict(X_test)

print("\nActual vs Predicted Values:")

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(result.head(10))



mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


result.to_csv(
    "student_math_predictions.csv",
    index=False
)

print("\nPrediction file saved successfully!")

print("\n====================================")
print("REGRESSION COMPLETED SUCCESSFULLY")
print("====================================")


import pandas as pd

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first 5 rows
print(data.head())

# Display dataset shape
print("Shape:", data.shape)

# Display column names
print(data.columns)
# Remove Customer ID
data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Fill missing values
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# Feature Engineering
data["AvgMonthlySpend"] = data["TotalCharges"] / (data["tenure"] + 1)

print(data.head())
print(data.shape)
# Convert categorical columns into numerical
data = pd.get_dummies(data, drop_first=True)

print(data.head())
print(data.shape)
from sklearn.model_selection import train_test_split

# Features and Target
X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
import joblib

# Save the trained model
joblib.dump(rf_model, "customer_churn_model.pkl")

print("Model saved successfully!")
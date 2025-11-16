import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/ibm_hr_attrition.csv")

# Encode 'Attrition' target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Define the features you want to use for the model
# These should be the same as in your app.py
features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
            'EnvironmentSatisfaction', 'TrainingTimesLastYear']

# If 'OverTime' is a relevant feature you want to use,
# ensure it's processed consistently in both training and prediction.
# For simplicity, let's assume you want to use the same features you listed in app.py for now.

# Handle categorical features within the selected subset if they exist
# In your chosen subset, 'JobSatisfaction' and 'EnvironmentSatisfaction' might be numerical already,
# but if any other categorical features were added, you'd encode them here.
# For the current list, they are likely already numeric or ordinal.

# Create X and y with *only* the desired features
X = df[features].copy() # Use .copy() to avoid SettingWithCopyWarning
y = df["Attrition"]

# Handle any NaN values that might be in the selected features
X = X.fillna(0) # As you do in app.py

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42) # Use 0.0 for full data training if you want, or keep 0.25 for testing
# Note: For a final model, you might train on all data. For evaluation, use a split.
# Let's keep a split for robustness.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional for a production model, but good for verification)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "outputs/rf_attrition.joblib")
print("✅ Model saved to outputs/rf_attrition.joblib")
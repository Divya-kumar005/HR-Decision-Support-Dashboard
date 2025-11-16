# src/ml_model.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder # Needed if you have other object columns not handled by prepare_features

def prepare_features(df_input): # Renamed to avoid confusion with internal df
    df = df_input.copy() # Work on a copy to avoid modifying the original df
    # Select features
    X = df[['Age','MonthlyIncome','YearsAtCompany','JobSatisfaction','EnvironmentSatisfaction','TrainingTimesLastYear']]

    # If OverTime is present, add OverTimeFlag
    if 'OverTime' in df.columns:
        X['OverTimeFlag'] = df['OverTime'].apply(lambda x: 1 if str(x).strip().lower()=='yes' else 0)
    else:
        # If OverTime is not in the dataframe, and the model was trained with OverTimeFlag,
        # you need to decide how to handle this. For now, let's assume it's set to 0.
        # This needs to be consistent with how missing 'OverTime' was handled in training data.
        if 'OverTimeFlag' in clf.feature_names_in_: # If the model expects it, but data doesn't have source
            X['OverTimeFlag'] = 0 # Or some default value.
            # This logic needs to be robust, ideally ensuring all needed raw columns are present.


    # Fill NaNs in the selected features
    X = X.fillna(0) # Ensure consistency with training
    return X

def train_random_forest(data_path="data/ibm_hr_attrition.csv", model_out="outputs/rf_attrition.joblib"):
    df = pd.read_csv(data_path)

    # Encode 'Attrition' to 'AttritionFlag' for the target variable
    df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Prepare features using the defined function
    X = prepare_features(df)
    y = df['AttritionFlag'] # Ensure this target column exists

    # Train-test split
    # Using stratify=y is good for imbalanced datasets like attrition
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate (optional, but good for understanding model performance)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Save the trained model
    joblib.dump(clf, model_out)
    print(f"✅ Model saved to {model_out}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {auc:.2f}")

    return clf, report, auc, cm

# Example of how to run this script (e.g., in a notebook or directly)
if __name__ == "__main__":
    # Ensure the outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    train_random_forest()
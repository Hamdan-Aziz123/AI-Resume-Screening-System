import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def safe_eval_list(x):
    """Safely evaluate string representation of a list."""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

def load_and_preprocess_data(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Safely process 'Skills' and 'Certifications'
    df['Skills'] = df['Skills'].apply(lambda x: safe_eval_list(x) if pd.notna(x) else [])
    df['Certifications'] = df['Certifications'].apply(lambda x: safe_eval_list(x) if pd.notna(x) else [])

    # Feature engineering
    df['Skills_Count'] = df['Skills'].apply(len)
    df['Has_Certifications'] = df['Certifications'].apply(lambda x: 1 if len(x) > 0 else 0)

    # Define features and target
    X = df[['Skills_Count', 'Experience (Years)', 'Education',
            'Has_Certifications', 'Projects Count', 'Job Role']]
    y = df['AI Score (0-100)']

    return X, y

def build_model_pipeline():
    # Define categorical and numerical features
    categorical_features = ['Education', 'Job Role']
    numerical_features = ['Skills_Count', 'Experience (Years)', 'Has_Certifications', 'Projects Count']

    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return pipeline

def train_and_save_model(csv_path, model_save_path='resume_scorer.joblib'):
    # Load and preprocess data
    X, y = load_and_preprocess_data(csv_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_model_pipeline()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluation - Training
    train_r2 = model.score(X_train, y_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    # Evaluation - Testing
    test_r2 = model.score(X_test, y_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    # Print Evaluation Metrics
    print("\n--- Training Metrics ---")
    print(f"R² Score: {train_r2:.3f}")
    print(f"MAE: {train_mae:.3f}")
    print(f"MSE: {train_mse:.3f}")
    print(f"RMSE: {train_rmse:.3f}")

    print("\n--- Testing Metrics ---")
    print(f"R² Score: {test_r2:.3f}")
    print(f"MAE: {test_mae:.3f}")
    print(f"MSE: {test_mse:.3f}")
    print(f"RMSE: {test_rmse:.3f}")

    # Save model
    joblib.dump(model, model_save_path)
    print(f"\nModel saved to {model_save_path}")

    return model

if __name__ == "__main__":
    train_and_save_model('AI_Resume_Screening.csv')

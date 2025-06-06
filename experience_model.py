import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib

df = pd.read_csv('experience_level_dataset.csv')

print(df.head())

X = df[['Experience (Years)', 'Projects Count', 'Certifications Count', 'Education', 'Skills Count', 'Resume Pages', 'Job Role']]
y = df['Experience Level']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

numerical_features = ['Experience (Years)', 'Projects Count', 'Certifications Count', 'Skills Count', 'Resume Pages']
categorical_features = ['Education', 'Job Role']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Training Accuracy: {model_pipeline.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print("\nDetailed Classification Report:")
print(report)

joblib.dump(model_pipeline, 'experience_level_model.joblib')
joblib.dump(label_encoder, 'experience_level_label_encoder.joblib')

def predict_experience_level(experience_years, projects_count, certifications_count, education, skills_count, resume_pages, job_role):
    input_data = pd.DataFrame([[experience_years, projects_count, certifications_count, education, skills_count, resume_pages, job_role]],
                              columns=['Experience (Years)', 'Projects Count', 'Certifications Count', 'Education', 'Skills Count', 'Resume Pages', 'Job Role'])
    predicted_label = model_pipeline.predict(input_data)
    experience_level = label_encoder.inverse_transform(predicted_label)[0]
    return experience_level

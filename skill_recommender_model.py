import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
df = pd.read_csv("domain_skills.csv")
X = df[['Field']]
y = df.drop(columns=['Field'])

# Encode the field (categorical input)
preprocessor = ColumnTransformer(
    transformers=[
        ('field', OneHotEncoder(), ['Field'])
    ]
)

# Create model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

# Train the model
model_pipeline.fit(X, y)

# Save the model
joblib.dump(model_pipeline, 'skill_recommender_model.joblib')
print("Model trained and saved!")

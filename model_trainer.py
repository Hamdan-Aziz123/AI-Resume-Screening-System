import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("resume_data.csv")

df['skills'] = df['skills'].apply(lambda x: x.replace(',', ' ') if isinstance(x, str) else '')

cv = CountVectorizer()
X = cv.fit_transform(df['skills'])

y = df['field']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "career_model.pkl")
joblib.dump(cv, "vectorizer.pkl")

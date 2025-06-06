import random
import joblib
import pandas as pd
from typing import List, Tuple
from Courses import ds_course, web_course, android_course, ios_course, uiux_course

class CareerRecommender:
    def __init__(self):
        self.model = joblib.load("career_model.pkl")
        self.vectorizer = joblib.load("vectorizer.pkl")
        self.model1 = joblib.load('skill_recommender_model.joblib')
        self.skills = pd.read_csv("domain_skills.csv").drop(columns=['Field']).columns.tolist()

    def get_recommended_skills(self, field: str) -> List[str]:
        """Predict recommended skills for a given career field"""
        input_df = pd.DataFrame({'Field': [field]})
        prediction = self.model1.predict(input_df)[0]  # array of 0/1s

        recommended_skills = [
            skill for skill, present in zip(self.skills, prediction) if present
        ]
        return recommended_skills
    
    def predict_field(self, skills: List[str]) -> str:
        """Predict career field based on skills"""
        skills_text = ' '.join(skills)
        vect_text = self.vectorizer.transform([skills_text])
        return self.model.predict(vect_text)[0]
    
    def recommend_courses(self, field: str, num_recommendations: int = 5) -> List[Tuple[str, str]]:
        """Recommend courses based on career field"""
        course_map = {
            'Data Science': ds_course,
            'Web Development': web_course,
            'Android Development': android_course,
            'IOS Development': ios_course,
            'UI-UX Development': uiux_course
        }
        courses = course_map.get(field, [])
        random.shuffle(courses)
        return courses[:num_recommendations]
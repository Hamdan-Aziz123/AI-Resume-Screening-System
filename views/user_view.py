import streamlit as st
import time
from typing import Dict, List, Tuple
from resume_parser import parse_resume, pdf_reader, show_pdf
from recommender import CareerRecommender
from utils import get_timestamp, prepare_data_for_db
from streamlit_tags import st_tags
from database import Database
import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import random
from ai_services.cover_letter_generator import CoverLetterGenerator

class UserView:
    def __init__(self):
        self.recommender = CareerRecommender()
        self.db = Database()
        self.resume_scorer = self._load_scoring_model()
        self.experience_model = self._load_experience_model()
        self.label_encoder = self._load_label_encoder()
    
    def show_upload_section(self):
        """Show resume upload section"""
        st.subheader("📤 Upload Your Resume")
        return st.file_uploader("Choose your Resume (PDF only)", type=["pdf"], accept_multiple_files=False)
    
    def analyze_resume(self, pdf_file, save_path):
        """Analyze uploaded resume"""
        with st.spinner('Analyzing your resume...'):
            time.sleep(2)
            
            # Save uploaded file
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Show PDF Preview
            st.subheader("📄 Resume Preview")
            st.markdown(show_pdf(save_path), unsafe_allow_html=True)
            
            # Parse resume data
            resume_data = parse_resume(save_path)
            if not resume_data:
                st.error("Failed to parse your resume. Please try another file.")
                return None
            
            resume_text = pdf_reader(save_path)
            return resume_data, resume_text
    
    def show_personal_info(self, resume_data: Dict):
        """Display personal information section"""
        st.success(f"Hello {resume_data['name']}!")
        with st.expander("👤 Personal Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Name", value=resume_data.get('name', ''), disabled=True)
                st.text_input("Email", value=resume_data.get('email', ''), disabled=True)
            with col2:
                st.text_input("Contact", value=resume_data.get('mobile_number', ''), disabled=True)
                pages = str(resume_data.get('no_of_pages', '')) + " pages"
                st.text_input("Pages", value=pages, disabled=True)
                
    def _load_experience_model(self):
        """Load the trained experience level model"""
        try:
            model = joblib.load('experience_level_model.joblib')
            print("Experience level model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading experience model: {str(e)}")
            return None
    
    def _load_label_encoder(self):
        """Load the label encoder for experience levels"""
        try:
            encoder = joblib.load('experience_level_label_encoder.joblib')
            return encoder
        except Exception as e:
            print(f"Error loading label encoder: {str(e)}")
            return None
    
    def determine_experience_level(self, resume_data: Dict, resume_text: str) -> str:
        """Predict experience level using ML model"""
        if not self.experience_model or not self.label_encoder:
            return self._fallback_experience_level(resume_data['no_of_pages'])
        
        try:
            # Extract features from resume
            features = self._prepare_experience_features(resume_data, resume_text)
            
            # Create DataFrame for prediction
            feature_df = pd.DataFrame([features], columns=[
                'Experience (Years)', 'Projects Count', 'Certifications Count',
                'Education', 'Skills Count', 'Resume Pages', 'Job Role'
            ])
            
            # Predict experience level
            predicted_label = self.experience_model.predict(feature_df)[0]
            experience_level = self.label_encoder.inverse_transform([predicted_label])[0]
            
            # Display result
            st.info(f"You are at **{experience_level}** level!")
            return experience_level
            
        except Exception as e:
            print(f"Experience prediction failed: {str(e)}")
            return self._fallback_experience_level(resume_data['no_of_pages'])

    def _prepare_experience_features(self, resume_data: Dict, resume_text: str) -> dict:
        """Prepare features for experience level prediction"""
        return {
            'Experience (Years)': self._estimate_experience_years(resume_text),
            'Projects Count': self._count_projects(resume_text),
            'Certifications Count': len(resume_data.get('certifications', [])),
            'Education': self._extract_education_level(resume_text),
            'Skills Count': len(resume_data.get('skills', [])),
            'Resume Pages': resume_data.get('no_of_pages', 1),
            'Job Role': resume_data.get('reco_field', 'Unknown')
        }

    def _estimate_experience_years(self, text: str) -> float:
        """Estimate years of experience from resume text"""
        # Look for explicit experience mentions
        experience_phrases = [
            'years of experience', 'years experience', 'yr exp',
            'experience of', 'worked for'
        ]
        
        for phrase in experience_phrases:
            if phrase in text.lower():
                # Try to extract the number
                try:
                    words = text.lower().split()
                    idx = words.index(phrase.split()[0])
                    if idx > 0 and words[idx-1].isdigit():
                        return float(words[idx-1])
                except:
                    continue
        
        # Fallback: estimate based on content density
        experience_keywords = [
            'company', 'position', 'role', 'responsibilities',
            'employed', 'worked', 'job'
        ]
        count = sum(text.lower().count(keyword) for keyword in experience_keywords)
        return min(count / 2.0, 15)  # Cap at 15 years

    def _count_projects(self, text: str) -> int:
        """Count mentioned projects in resume text"""
        return text.lower().count('project')

    def _extract_education_level(self, text: str) -> str:
        """Extract highest education level from text"""
        text_lower = text.lower()
        if 'phd' in text_lower or 'doctorate' in text_lower:
            return 'PhD'
        elif 'master' in text_lower or 'msc' in text_lower or 'mba' in text_lower:
            return 'Masters'
        elif 'bachelor' in text_lower or 'bsc' in text_lower or 'undergraduate' in text_lower:
            return 'Bachelors'
        elif 'diploma' in text_lower:
            return 'Diploma'
        return 'Unknown'

    def _fallback_experience_level(self, num_pages: int) -> str:
        """Fallback method if model fails"""
        if num_pages == 1:
            st.info("You are at **Fresher** level!")
            return "Fresher"
        elif num_pages == 2:
            st.info("You are at **Intermediate** level!")
            return "Intermediate"
        else:
            st.info("You are at **Experienced** level!")
            return "Experienced"
    
    def show_skills_analysis(self, skills: List[str]):
        """Display skills analysis section"""
        st.subheader("🔍 Skills Analysis")
        with st.expander("Your Current Skills"):
            if skills:
                st_tags(
                    label='### Your Current Skills',
                    text='See our skills recommendation below',
                    value=skills,
                    key='1'
                )
                st.write("Extracted Skills:", skills)
                st.markdown("**These skills are essential for your career growth!**")
                return skills
            return []
                
    
    def show_career_prediction(self, skills: List[str]):
        """Display career prediction section"""
        st.subheader("🔮 Career Field Prediction")
        reco_field = self.recommender.predict_field(skills)
        st.success(f"Our AI model predicts you're best suited for: **{reco_field}**")
        return reco_field
    
    def show_recommended_skills(self, field: str):
        """Display recommended skills section"""
        st.subheader("🚀 Recommended Skills")
        recommended_skills = self.recommender.get_recommended_skills(field)
        
        if recommended_skills:
            rec_skills = st_tags(
                label='### Recommended Skills for Career Growth',
                text='Based on your profile and industry trends',
                value=recommended_skills,
                key='2'
            )
            st.markdown("**Adding these skills will boost your career prospects!**")
            return recommended_skills
        return []
    
    def show_course_recommendations(self, field: str, num_recommendations: int = 5):
        """Display course recommendations"""
        st.subheader("🎓 Learning Recommendations")
        courses = self.recommender.recommend_courses(field, num_recommendations)
        
        for i, (name, link) in enumerate(courses, 1):
            st.markdown(f"({i}) [{name}]({link})")
        
        return courses
    
    def _load_scoring_model(self):
        """Load the trained resume scoring model"""
        try:
            model = joblib.load('resume_scorer.joblib')
            print("Resume scoring model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading scoring model: {str(e)}")
            return None
    
    def prepare_features(self, resume_data: dict, resume_text: str) -> dict:
        """Prepare features for the scoring model"""
        features = {
            'Skills_Count': len(resume_data.get('skills', [])),
            'Experience (Years)': self._estimate_experience(resume_text),
            'Education': self._extract_education(resume_text),
            'Has_Certifications': int('certifications' in resume_text.lower()),
            'Projects Count': self._count_projects(resume_text),
            'Job Role': resume_data.get('reco_field', 'Unknown')
        }
        return features
    
    def _estimate_experience(self, text: str) -> float:
        """Estimate years of experience from resume text"""
        # Simple implementation - count experience-related phrases
        exp_phrases = ['years of experience', 'yr exp', 'experience']
        count = sum(text.lower().count(phrase) for phrase in exp_phrases)
        return min(count, 15)  # Cap at 15 years
    
    def _extract_education(self, text: str) -> str:
        """Extract highest education level"""
        text_lower = text.lower()
        if 'phd' in text_lower or 'doctorate' in text_lower:
            return 'PhD'
        elif 'master' in text_lower:
            return 'Masters'
        elif 'bachelor' in text_lower or 'undergraduate' in text_lower:
            return 'Bachelors'
        elif 'diploma' in text_lower:
            return 'Diploma'
        return 'Unknown'
    
    def _count_projects(self, text: str) -> int:
        """Count mentioned projects"""
        return text.lower().count('project')
    
    def calculate_resume_score(self, resume_text: str, resume_data: dict) -> int:
        """Calculate resume score using trained model"""
        try:
            if not self.resume_scorer:
                raise ValueError("Scoring model not available")
            
            # Prepare features
            features = self.prepare_features(resume_data, resume_text)
            feature_df = pd.DataFrame([features])
            
            # Predict score
            predicted_score = self.resume_scorer.predict(feature_df)[0]
            predicted_score = int(np.clip(predicted_score, 0, 100))
            
            # Display results
            self._display_score(predicted_score)
            
            return predicted_score
            
        except Exception as e:
            print(f"Score prediction failed: {str(e)}")
            return self.fallback_score_calculation(resume_text)
    
    def _display_score(self, score: int):
        """Display the score with visual feedback"""
        st.progress(score/100)
        st.metric("AI Resume Score", f"{score}/100")
        
        if score < 50:
            st.error("Needs significant improvement")
            st.markdown("""
            **Recommendations:**
            - Add more relevant skills
            - Highlight key projects
            - Include certifications
            - Improve education section
            """)
        elif score < 75:
            st.warning("Good but could be improved")
            st.markdown("""
            **Suggestions:**
            - Quantify achievements
            - Add more technical details
            - Include measurable results
            """)
        else:
            st.success("Excellent resume!")
            st.markdown("""
            **Great job!** Your resume scores well on:
            - Skills relevance
            - Experience demonstration
            - Overall completeness
            """)
    
    def fallback_score_calculation(self, resume_text: str) -> int:
        """Fallback scoring method"""
        score = 0
        lower_text = resume_text.lower()
        
        if any(word in lower_text for word in ['objective', 'objectives']):
            score += 15
        if 'declaration' in lower_text:
            score += 10
        if any(word in lower_text for word in ['hobbies', 'interests']):
            score += 10
        if 'achievements' in lower_text:
            score += 20
        if 'projects' in lower_text:
            score += 20
        if any(word in lower_text for word in ['education', 'degree']):
            score += 15
        if any(word in lower_text for word in ['experience', 'work history']):
            score += 10
            
        self._display_score_results(score)
        return score

    
    def save_to_database(self, resume_data: Dict, resume_score: int, reco_field: str, 
                        cand_level: str, recommended_skills: List[str], courses: List[Tuple[str, str]]):
        """Save candidate data to database"""
        timestamp = get_timestamp()
        db_data = prepare_data_for_db(
            resume_data['name'],
            resume_data['email'],
            resume_score,
            timestamp,
            resume_data['no_of_pages'],
            reco_field,
            cand_level,
            resume_data['skills'],
            recommended_skills,
            courses
        )
        self.db.insert_candidate_data(db_data)
    
    def show_career_resources(self, resume_videos, interview_videos):
        """Display career resources section"""
        tab_res, tab_int = st.tabs(["Resume Tips", "Interview Tips"])
        
        with tab_res:
            st.write("**Watch these videos to improve your resume:**")
            resume_vid = random.choice(resume_videos)
            st.video(resume_vid)
        
        with tab_int:
            st.write("**Prepare for interviews with these tips:**")
            interview_vid = random.choice(interview_videos)
            st.video(interview_vid)
        
        st.balloons()
        
    def show_cover_letter_tool(self, resume_data, cover_letter_generator, resume_text, reco_field, recommended_skills):
        with st.expander("Generate Tailored Cover Letter"):
            company = st.text_input("Target Company")
            job_title = st.text_input("Job Title")
            job_desc = st.text_area("Paste Job Description")
            
            if st.button("Generate Cover Letter"):
                if company and job_title:
                    with st.spinner('Generating your cover letter...'):
                        try:
                            letter = cover_letter_generator.generate(
                                resume_data={
                                    "name": resume_data.get("name", ""),
                                    "email": resume_data.get("email", ""),
                                    "skills": resume_data.get("skills", []),
                                    "summary": resume_text,
                                    "experience_level": self.determine_experience_level(resume_data['no_of_pages']),
                                    "recommended_skills": recommended_skills
                                },
                                job_description={
                                    "company": company,
                                    "title": job_title,
                                    "description": job_desc,
                                    "field": reco_field
                                }
                            )
                            st.text_area("Generated Cover Letter", letter, height=400)
                            st.success("Cover letter generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating cover letter: {str(e)}")
                else:
                    st.warning("Please enter company and job title")
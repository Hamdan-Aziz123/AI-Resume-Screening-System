import streamlit as st
import pandas as pd
import plotly.express as px
import re
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from database import Database
from config import ADMIN_CREDENTIALS
from utils import get_table_download_link
import numpy as np

class AdminView:
    def __init__(self):
        self.db = Database()
    
    def show_login(self):
        """Show admin login form"""
        st.subheader("Admin Login")
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            
            if st.form_submit_button("Login"):
                if username == ADMIN_CREDENTIALS['username'] and password == ADMIN_CREDENTIALS['password']:
                    st.session_state.admin_logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Wrong credentials")
    
    def show_data_overview(self):
        """Show data overview section"""
        st.subheader("📊 Data Overview")
        columns, data = self.db.get_all_candidates()
        df = pd.DataFrame(data, columns=columns)
        
        # Quick Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Candidates", len(df))
        with col2:
            avg_score = df['resume_score'].astype(float).mean().round(1)
            st.metric("Average Resume Score", avg_score)
        with col3:
            st.metric("Most Common Field", df['Predicted_Field'].mode()[0])
        
        return df
    
    def show_analytics_dashboard(self, df):
        """Show analytics dashboard"""
        st.subheader("📈 Analytics Dashboard")
        
        viz_tab1, viz_tab2 = st.tabs(["Field Distribution", "Experience Levels"])
        
        with viz_tab1:
            field_counts = df['Predicted_Field'].value_counts()
            fig = px.pie(
                names=field_counts.index.tolist(),
                values=field_counts.values.tolist(),
                title='Candidate Distribution by Predicted Field')
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            level_counts = df['User_level'].value_counts()
            fig2 = px.pie(
                names=level_counts.index.tolist(),
                values=level_counts.values.tolist(),
                title="Candidate Experience Levels"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    def show_candidate_database(self, df):
        """Show candidate database section"""
        st.subheader("📋 Candidate Database")
        st.dataframe(df)
        st.markdown(get_table_download_link(df, 'User_Data.csv', '📥 Download Full Data'), unsafe_allow_html=True)
    
    def _prepare_resume_data(self, df):
        """Prepare resume data for screening"""
        resumes = df[['Name', 'Email_ID', 'Actual_skills', 'Predicted_Field']].copy()
        
        # Convert skills to strings
        resumes['skills_str'] = resumes['Actual_skills'].apply(
            lambda x: ' '.join(literal_eval(x)) if isinstance(x, str) else ' '.join(x)
        )
        
        # Add other relevant fields if available
        if 'resume_text' in df.columns:
            resumes['full_text'] = df['resume_text']
        else:
            resumes['full_text'] = resumes['skills_str']
            
        return resumes
    
    def show_screening_tool(self, df):
        """Show AI screening tool using advanced NLP"""
        st.subheader("🔍 AI Resume Screening")
        
        job_description = st.text_area(
            "Paste Job Description (e.g., from LinkedIn):",
            height=200,
            help="The more detailed the job description, the better the matching results will be."
        )
 
        with st.expander("⚙️ Advanced Options"):
            model_name = st.selectbox(
                "Matching Model",
                options=[
                    'all-MiniLM-L6-v2',  
                    'all-mpnet-base-v2',  
                    'multi-qa-MiniLM-L6-cos-v1'  
                ],
                index=0
            )
            
            match_threshold = st.slider(
                "Minimum Match Score (%)",
                min_value=0,
                max_value=100,
                value=50,
                help="Only show candidates with match scores above this threshold"
            )
        
        if job_description:
            @st.cache_resource
            def load_model(model_name):
                return SentenceTransformer(model_name)
            
            model = load_model(model_name)
            resumes = self._prepare_resume_data(df)
 
            with st.spinner("Analyzing job description and resumes..."):
                job_embedding = model.encode([job_description])
                resume_embeddings = model.encode(resumes['full_text'].tolist())
      
                similarity_scores = cosine_similarity(job_embedding, resume_embeddings).flatten()
   
                resumes['Match Score (%)'] = (similarity_scores * 100).round(2)
                ranked_resumes = resumes.sort_values(by='Match Score (%)', ascending=False)
                ranked_resumes = ranked_resumes[ranked_resumes['Match Score (%)'] >= match_threshold]
                ranked_resumes = ranked_resumes.reset_index(drop=True)
   
            st.subheader("🏆 Matching Candidates")
            
            if len(ranked_resumes) == 0:
                st.warning("No candidates meet the minimum match threshold. Try adjusting the threshold or using a more detailed job description.")
            else:
                st.dataframe(
                    ranked_resumes[['Name', 'Email_ID', 'Predicted_Field', 'Match Score (%)']],
                    height=min(400, 35 * (len(ranked_resumes) + 1))
                )
            
                fig = px.bar(
                    ranked_resumes.head(20),
                    x='Name',
                    y='Match Score (%)',
                    color='Predicted_Field',
                    title=f'Top Candidate Matches (Threshold: {match_threshold}%)',
                    hover_data=['Email_ID']
                )
                st.plotly_chart(fig, use_container_width=True)
 
                if not ranked_resumes.empty:
                    top_candidate = ranked_resumes.iloc[0]
                    
                    st.subheader("🎯 Best Match Details")
                    cols = st.columns([1, 1, 2])
                    
                    with cols[0]:
                        st.markdown("**Candidate Info**")
                        st.markdown(f"**Name:** {top_candidate['Name']}")
                        st.markdown(f"**Email:** {top_candidate['Email_ID']}")
                        st.markdown(f"**Field:** {top_candidate['Predicted_Field']}")
                        st.markdown(f"**Match Score:** {top_candidate['Match Score (%)']}%")
                    
                    with cols[1]:
                        st.markdown("**Key Skills**")
                        skills = literal_eval(top_candidate['Actual_skills']) if isinstance(top_candidate['Actual_skills'], str) else top_candidate['Actual_skills']
                        for skill in skills[:10]:  
                            st.markdown(f"- {skill}")
                    
                    with cols[2]:
                        st.markdown("**Match Analysis**")
                        st.markdown("**Relevant Skills:**")
                        skills_text = ' '.join(skills)
                        job_keywords = set(job_description.lower().split())
                        relevant_skills = [s for s in skills if any(kw in s.lower() for kw in job_keywords)]
                        
                        if relevant_skills:
                            for skill in relevant_skills[:5]:
                                st.markdown(f"- {skill}")
                        else:
                            st.info("No exact keyword matches found - matching based on semantic similarity")
    
    def show_logout(self):
        """Show logout button"""
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.success("Logged out successfully!")
            st.rerun()
import streamlit as st
from PIL import Image
from views.user_view import UserView
from views.admin_view import AdminView
from Courses import resume_videos, interview_videos
from config import UPLOAD_FOLDER
from ai_services.cover_letter_generator import CoverLetterGenerator

st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="🧠",
    layout="wide"
)

user_view = UserView()
admin_view = AdminView()
cover_letter_generator = CoverLetterGenerator()

col1, col2 = st.columns([1, 3])
with col1:
    logo = Image.open('./Logo/Logo Ai.png')  
    st.image(logo, width=150)
with col2:
    st.title("AI Resume Analyzer Pro")
    st.caption("Your AI-powered career companion")

tab1, tab2, tab3 = st.tabs(["🏠 Home", "👤 User Dashboard", "🔒 Admin Dashboard"])

with tab1:
    st.title("👋 Welcome to AI Resume Analyzer Pro")
    st.subheader("Your AI-powered career companion.")
    st.markdown("""
    - Upload your resume  
    - Get job field predictions  
    - Analyze skills  
    - Generate cover letters  
    - Access learning resources
    """)
    st.image("https://img.freepik.com/free-vector/job-interview-concept-illustration_114360-1545.jpg", use_container_width=True)

with tab2:
    st.title("📄 Resume Analysis Portal")

    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. Upload your resume (PDF)  
        2. We extract and analyze your skills  
        3. Get personalized field prediction  
        4. Receive skill & course recommendations  
        5. Generate a personalized cover letter  
        """)

    pdf_file = user_view.show_upload_section()
    if pdf_file:
        save_path = f"{UPLOAD_FOLDER}{pdf_file.name}"
        result = user_view.analyze_resume(pdf_file, save_path)

        if result:
            resume_data, resume_text = result
            user_view.show_personal_info(resume_data)
            cand_level = user_view.determine_experience_level(resume_data, resume_text)

            st.subheader("🧠 Skills & Career Insights")
            user_view.show_skills_analysis(resume_data['skills'])
            reco_field = user_view.show_career_prediction(resume_data['skills'])
            recommended_skills = user_view.show_recommended_skills(reco_field)
            courses = user_view.show_course_recommendations(reco_field)
            resume_score = user_view.calculate_resume_score(resume_text, resume_data)

            user_view.save_to_database(
                resume_data, resume_score, reco_field,
                cand_level, recommended_skills, courses
            )

            st.subheader("📝 Cover Letter Generator")
            user_view.show_cover_letter_tool(
                resume_data, cover_letter_generator,
                resume_text, reco_field, recommended_skills
            )

            st.subheader("🎥 Career Resources")
            user_view.show_career_resources(resume_videos, interview_videos)

with tab3:
    st.title("🔒 Admin Dashboard")

    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        admin_view.show_login()
    else:
        st.success("Welcome, Admin!")
        df = admin_view.show_data_overview()
        admin_view.show_analytics_dashboard(df)
        admin_view.show_candidate_database(df)
        admin_view.show_screening_tool(df)
        admin_view.show_logout()

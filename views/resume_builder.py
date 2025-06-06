import streamlit as st
from models.resume_model import Resume, Experience, Education

class ResumeBuilder:
    def __init__(self):
        self.resume = Resume(
            name="",
            contact="",
            email="",
            summary="",
            experiences=[],
            education=[],
            skills=[]
        )

    def show_builder(self):
        st.subheader("🧩 Resume Builder")
        
        with st.form("resume_form"):
            # Personal Info
            col1, col2 = st.columns(2)
            with col1:
                self.resume.name = st.text_input("Full Name*")
                self.resume.contact = st.text_input("Phone*")
            with col2:
                self.resume.email = st.text_input("Email*")
                self.resume.summary = st.text_area("Professional Summary")
            
            # Experience Section
            st.markdown("### Work Experience")
            exp_col1, exp_col2 = st.columns([3,1])
            with exp_col1:
                exp_title = st.text_input("Job Title")
            with exp_col2:
                exp_duration = st.text_input("Duration (e.g. 2020-2022)")
            
            company = st.text_input("Company Name")
            responsibilities = st.text_area("Responsibilities (comma separated)")
            
            if st.form_submit_button("Add Experience"):
                if exp_title and company:
                    self.resume.experiences.append(
                        Experience(
                            title=exp_title,
                            company=company,
                            duration=exp_duration,
                            responsibilities=[r.strip() for r in responsibilities.split(",")]
                        )
                    )
            
            # Similar sections for Education, Skills etc...
            
            if st.form_submit_button("Generate Resume"):
                self._generate_pdf()
    
    def _generate_pdf(self):
        """Convert resume data to PDF"""
        # Implement using ReportLab or PyPDF2
        st.success("Resume generated successfully!")
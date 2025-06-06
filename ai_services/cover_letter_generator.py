from openai import OpenAI
from config import OPENAI_API_KEY

class CoverLetterGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.template = """Write a professional cover letter for {name} applying for the {job_title} position at {company}.
        
        Candidate Background:
        - Experience Level: {experience_level}
        - Key Skills: {skills}
        - Additional Recommended Skills: {recommended_skills}
        
        Job Requirements:
        {job_description}
        
        The letter should:
        1. Be addressed to the hiring manager
        2. Highlight relevant skills and experience
        3. Show enthusiasm for the role and company
        4. Be concise (under 400 words)
        5. Use professional business letter format
        """
    
    def generate(self, resume_data: dict, job_description: dict) -> str:
        prompt = self.template.format(
            name=resume_data.get("name", "the candidate"),
            job_title=job_description.get("title", ""),
            company=job_description.get("company", ""),
            experience_level=resume_data.get("experience_level", ""),
            skills=", ".join(resume_data.get("skills", [])),
            recommended_skills=", ".join(resume_data.get("recommended_skills", [])),
            job_description=job_description.get("description", "")
        )
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
from typing import List, Dict, Optional
from pydantic import BaseModel

class Experience(BaseModel):
    title: str
    company: str
    duration: str
    responsibilities: List[str]

class Education(BaseModel):
    degree: str
    institution: str
    year: str

class Resume(BaseModel):
    name: str
    contact: str
    email: str
    summary: str
    experiences: List[Experience]
    education: List[Education]
    skills: List[str]
    projects: Optional[List[str]]
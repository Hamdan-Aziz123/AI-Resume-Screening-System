import os
from dotenv import load_dotenv 

load_dotenv()

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': 'cv',
    'port': 3306
}

UPLOAD_FOLDER = './Uploaded_Resumes/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ADMIN_CREDENTIALS = {
    'username': 'hamid',
    'password': 'hamid123'
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
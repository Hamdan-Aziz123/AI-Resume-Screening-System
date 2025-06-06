# utils.py
import base64
import pandas as pd
import random
import time
import datetime
import json
from typing import List, Tuple

def get_table_download_link(df, filename, text):
    """Generate a download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def get_timestamp() -> str:
    """Generate current timestamp"""
    ts = time.time()
    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    return str(cur_date + '_' + cur_time)

def prepare_data_for_db(
    name: str,
    email: str,
    resume_score: str,
    timestamp: str,
    no_of_pages: str,
    reco_field: str,
    cand_level: str,
    skills: List[str],
    recommended_skills: List[str],
    courses: List[Tuple[str, str]]
) -> tuple:
    """Prepare data for database insertion"""
    return (
        name, email, str(resume_score), timestamp, str(no_of_pages),
        json.dumps(reco_field), json.dumps(cand_level), json.dumps(skills),
        json.dumps(recommended_skills), json.dumps(courses)
    )
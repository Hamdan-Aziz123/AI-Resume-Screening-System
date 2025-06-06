# database.py
import pymysql
from config import DB_CONFIG

class Database:
    def __init__(self):
        self.connection = pymysql.connect(**DB_CONFIG)
        self.cursor = self.connection.cursor()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database and tables"""
        self.cursor.execute("CREATE DATABASE IF NOT EXISTS CV;")
        
        table_sql = """CREATE TABLE IF NOT EXISTS user_data (
                        ID INT NOT NULL AUTO_INCREMENT,
                        Name varchar(500) NOT NULL,
                        Email_ID VARCHAR(500) NOT NULL,
                        resume_score VARCHAR(8) NOT NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        Page_no VARCHAR(5) NOT NULL,
                        Predicted_Field TEXT NOT NULL,
                        User_level TEXT NOT NULL,
                        Actual_skills TEXT NOT NULL,
                        Recommended_skills TEXT NOT NULL,
                        Recommended_courses TEXT NOT NULL,
                        PRIMARY KEY (ID));
                    """
        self.cursor.execute(table_sql)
        self.connection.commit()

    def insert_candidate_data(self, data):
        """Insert candidate data into database"""
        sql = """INSERT INTO user_data 
                VALUES (0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, data)
        self.connection.commit()

    def get_all_candidates(self):
        """Retrieve all candidate data"""
        self.cursor.execute('''SELECT * FROM user_data''')
        columns = [col[0] for col in self.cursor.description]
        data = self.cursor.fetchall()
        return columns, data

    def close(self):
        """Close database connection"""
        self.connection.close()
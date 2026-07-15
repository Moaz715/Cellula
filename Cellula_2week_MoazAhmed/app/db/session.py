import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv

# This function searches your project for the .env file and loads the variables
load_dotenv()

class SQLiteDatabaseManager:
    def __init__(self):
        # os.getenv pulls the variable. The second argument is a fallback just in case.
        self.db_path = os.getenv("DB_PATH", "api_logs.db")
        self._initialize_db()
        
    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    input_type TEXT,
                    content TEXT,
                    prediction TEXT
                )
            """)
            conn.commit()
            
    def log_entry(self, input_type: str, content: str, prediction: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO logs (timestamp, input_type, content, prediction) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), input_type, content, prediction)
            )
            conn.commit()
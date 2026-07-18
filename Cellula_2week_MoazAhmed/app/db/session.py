import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
import csv
import os
from datetime import datetime, UTC

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
            
            


class CSVDatabaseManager:
    def __init__(self, filename: str = "api_logs.csv"):
        self.filepath = os.path.join(os.getcwd(), filename)
        self.headers = ["timestamp", "input_type", "content", "prediction"]
        self._initialize_file()

    def _initialize_file(self):
        """Creates the CSV file and writes headers if it doesn't exist."""
        file_exists = os.path.isfile(self.filepath)
        
        if not file_exists:
            with open(self.filepath, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

    def log_entry(self, input_type: str, content: str, prediction: str):
        """Appends a single prediction event to the CSV."""
        timestamp = datetime.now(UTC).isoformat()
        
        clean_content = content.replace("\n", " ").replace("\r", " ")
        
        with open(self.filepath, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, input_type, clean_content, prediction])
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
import csv
import os
from datetime import datetime, UTC

load_dotenv()

class CSVDatabaseManager:
    def __init__(self):
        filename = os.getenv("CSV_LOG_FILE", "api_logs.csv")
        self.filepath = os.path.join(os.getcwd(), filename)
        self.headers = ["timestamp", "input_type", "content", "prediction"]
        self._initialize_file()

    def _initialize_file(self):
        file_exists = os.path.isfile(self.filepath)
        
        if not file_exists:
            with open(self.filepath, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

    def log_entry(self, input_type: str, content: str, prediction: str):
        timestamp = datetime.now(UTC).isoformat()
        
        clean_content = content.replace("\n", " ").replace("\r", " ")
        
        with open(self.filepath, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, input_type, clean_content, prediction])
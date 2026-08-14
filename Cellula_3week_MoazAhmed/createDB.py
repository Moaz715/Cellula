import sqlite3

def init_student_db():
    conn = sqlite3.connect("app_database.db")
    cursor = conn.cursor()

    #cursor.execute("DROP TABLE IF EXISTS students;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        gpa REAL NOT NULL
    );
    """)

    print("Inserting 15 sample student records...")
    
    dummy_students = [
        ("Mahmoud Ali", 2.9),
        ("Mona Hassan", 3.5),
        ("Mostafa Ibrahim", 3.1),
        ("Mariam Youssef", 4.0),
        
        ("Sara Kamal", 3.9),
        ("Samir Tarek", 2.5),
        ("Salma Fathy", 3.7),
        ("Seif Adel", 3.2),
        
        ("Ahmed Magdy", 2.8),
        ("Amira Hany", 3.6),
        ("Ali Omar", 3.4),
        ("Aya Nabil", 3.9),
        
        ("Youssef Zaki", 3.3),
        ("Nour Khaled", 3.8)
    ]
    
    cursor.executemany("""
    INSERT INTO students (name, gpa) VALUES (?, ?)
    """, dummy_students)

    conn.commit()
    conn.close()
    print("Database 'app_database.db' is ready with a fresh dataset!")

if __name__ == "__main__":
    init_student_db()
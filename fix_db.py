import sqlite3

def fix_db():
    conn = sqlite3.connect('surveillance.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE logs SET identity = REPLACE(identity, '(Unknown)', '(Known)') WHERE event_type = 'KNOWN'")
    conn.commit()
    conn.close()
    print("Database fixed successfully.")

if __name__ == '__main__':
    fix_db()

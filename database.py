import sqlite3
import os
import datetime
import threading

DB_PATH = "surveillance.db"
db_lock = threading.Lock()

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)

def init_db():
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Table for tracking logs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            identity TEXT NOT NULL,
            camera TEXT NOT NULL,
            confidence REAL,
            image_path TEXT
        )
    ''')
    
    # Table for stats (just a simple key-value store for total counts if needed, 
    # but we can also just compute them from the logs table + vector db)
        
        conn.commit()
        conn.close()

def log_event(event_type: str, identity: str, camera: str, confidence: float = None, image_path: str = None):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
        INSERT INTO logs (timestamp, event_type, identity, camera, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, event_type, identity, camera, confidence, image_path))
        conn.commit()
        conn.close()

def get_logs(limit: int = 50):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, event_type, identity, camera, confidence, image_path FROM logs ORDER BY id DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
    
    logs = []
    for row in rows:
        logs.append({
            "timestamp": row[0],
            "event_type": row[1],
            "identity": row[2],
            "camera": row[3],
            "confidence": row[4],
            "image_path": row[5]
        })
    return logs

def get_stats(total_known: int):
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM logs WHERE event_type = 'UNKNOWN'")
        total_unknown = cursor.fetchone()[0]
        
        conn.close()
    
    return {
        "active_cameras": 1,
        "known_identities": total_known,
        "unknown_alerts": total_unknown,
        "system_status": "Secure" if total_unknown == 0 else "Alert"
    }

# Initialize on import
init_db()

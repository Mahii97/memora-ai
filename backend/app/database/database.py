import sqlite3
from pathlib import Path

DB_PATH = Path("data/memora.db")

print("Using database:", DB_PATH.resolve())


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    conn = get_connection()
    cur = conn.cursor()

    # Documents table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id TEXT PRIMARY KEY,
        source_type TEXT,
        source_name TEXT,
        author TEXT,
        created_at TEXT,
        total_pages INTEGER,
        file_hash TEXT
    )
    """)

    # Safe migration for existing databases
    cur.execute("PRAGMA table_info(documents)")
    columns = [row[1] for row in cur.fetchall()]

    if "file_hash" not in columns:
        cur.execute("""
        ALTER TABLE documents
        ADD COLUMN file_hash TEXT
        """)

    # Original extracted content
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT,
        location TEXT,
        text TEXT
    )
    """)

    # Smart chunks
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks(
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT,
        source_type TEXT,
        source_name TEXT,
        author TEXT,
        created_at TEXT,
        location TEXT,
        chunk_index INTEGER,
        text TEXT,
        start_char INTEGER,
        end_char INTEGER
    )
    """)

    conn.commit()
    conn.close()
import sqlite3

def init_db(db_path="./zebra_verification.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS image_verification (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image1_path TEXT,
        image2_path TEXT,
        status TEXT CHECK(status IN ('awaiting', 'in_progress', 'correct', 'incorrect', 'cant_tell')) DEFAULT 'awaiting'
    )
    """)
    conn.commit()
    conn.close()


def add_image_pair(image1, image2, db_path="./zebra_verification.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO image_verification (image1_path, image2_path, status)
        VALUES (?, ?, 'awaiting')
    """, (image1, image2))
    conn.commit()
    print("Image pair added successfully.")
    conn.close()


def get_next_pair(db_path="./zebra_verification.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, image1_path, image2_path FROM image_verification
        WHERE status = 'awaiting'
        ORDER BY id ASC LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        cursor.execute("UPDATE image_verification SET status='in_progress' WHERE id=?", (result[0],))
        conn.commit()
    conn.close()
    return result


def update_status(id, new_status, db_path="./zebra_verification.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE image_verification SET status=? WHERE id=?", (new_status, id))
    conn.commit()
    conn.close()

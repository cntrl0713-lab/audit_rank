import sqlite3
import pandas as pd
import bcrypt
from datetime import datetime

DB_NAME = "audit_rank.db"

def init_db():
    """데이터베이스 및 테이블 초기화"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. 문제 풀이 기록 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            standard_code TEXT,
            score REAL,
            created_at TIMESTAMP
        )
    ''')
    
    # 2. 오답 노트 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS review_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            standard_code TEXT,
            question TEXT,
            answer TEXT,
            score REAL,
            created_at TIMESTAMP
        )
    ''')
    
    # 3. 사용자(등급) 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT DEFAULT 'MEMBER',
            level INTEGER DEFAULT 1,
            xp REAL DEFAULT 0,
            created_at TIMESTAMP
        )
    ''')
    
    # Migration: Recreate users table if 'password' column exists (from old schema)
    try:
        c.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in c.fetchall()]
        
        if 'password' in columns:
            # Rename old table
            c.execute("ALTER TABLE users RENAME TO users_old")
            
            # Create new table
            c.execute('''
                CREATE TABLE users (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    role TEXT DEFAULT 'MEMBER',
                    level INTEGER DEFAULT 1,
                    xp REAL DEFAULT 0,
                    created_at TIMESTAMP
                )
            ''')
            
            # Copy data
            c.execute("PRAGMA table_info(users_old)")
            old_columns = [info[1] for info in c.fetchall()]
            
            cols_to_copy = ['username']
            if 'role' in old_columns: cols_to_copy.append('role')
            if 'created_at' in old_columns: cols_to_copy.append('created_at')
            if 'level' in old_columns: cols_to_copy.append('level')
            if 'xp' in old_columns: cols_to_copy.append('xp')
            # If password existed in old, copy it. If not, it will be NULL (which is fine, they need to reset or we set default)
            if 'password' in old_columns: cols_to_copy.append('password')

            cols_str = ", ".join(cols_to_copy)
            c.execute(f"INSERT INTO users ({cols_str}) SELECT {cols_str} FROM users_old")
            
            # Drop old table
            c.execute("DROP TABLE users_old")
            
    except Exception as e:
        print(f"Migration warning: {e}")
        
    # Migration: Add password column if missing (for the case where we just removed it)
    try:
        c.execute("ALTER TABLE users ADD COLUMN password TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: Add level/xp columns if missing
    try:
        c.execute("ALTER TABLE users ADD COLUMN level INTEGER DEFAULT 1")
    except sqlite3.OperationalError: pass
    
    try:
        c.execute("ALTER TABLE users ADD COLUMN xp REAL DEFAULT 0")
    except sqlite3.OperationalError: pass

    # 초기 테스트 데이터 (관리자 및 게스트)
    # INSERT OR IGNORE: 이미 존재하면 무시
    # Note: For test users, we set a dummy password hash
    dummy_hash = bcrypt.hashpw(b'1234', bcrypt.gensalt())
    
    try:
        c.execute("INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)", 
              ('admin', dummy_hash, 'PRO', datetime.now()))
    except sqlite3.IntegrityError: pass
    
    try:
        c.execute("INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)", 
              ('guest', dummy_hash, 'GUEST', datetime.now()))
    except sqlite3.IntegrityError: pass
    
    conn.commit()
    conn.close()

def create_user(username, password):
    """Create a new user with hashed password."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Determine initial role
    role = 'MEMBER'
    if 'guest' in username.lower(): role = 'GUEST'
    elif 'admin' in username.lower(): role = 'ADMIN'
    
    try:
        c.execute("INSERT INTO users (username, password, role, level, xp, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                  (username, hashed, role, 1, 0, datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    """Validate user credentials."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("SELECT username, password, role, level, xp FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and user[1]:
        if bcrypt.checkpw(password.encode('utf-8'), user[1]):
            return user # (username, password, role, level, xp)
    return None

# ---------------------------------------------------------
# [사용자 및 등급 관리]
# ---------------------------------------------------------

def get_user_role(username):
    """사용자 ID를 받아 등급(Role)을 반환. 없으면 자동 가입 처리."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    
    if result:
        role = result[0]
    else:
        # 신규 유저 자동 등록 로직
        if 'guest' in username.lower():
            role = 'GUEST'
        elif 'admin' in username.lower(): # 테스트 편의상 admin 포함시 PRO 부여
            role = 'PRO'
        else:
            role = 'MEMBER'
            
        c.execute("INSERT INTO users (username, role, created_at) VALUES (?, ?, ?)", 
                  (username, role, datetime.now()))
        conn.commit()
    
    conn.close()
    return role

def get_user_stats(username):
    """사용자 통계 정보 조회"""
    conn = sqlite3.connect(DB_NAME)
    
    # 총점 및 풀이 수
    query_stats = "SELECT COUNT(*) as count, SUM(score) as total FROM quiz_history WHERE username = ?"
    df_stats = pd.read_sql_query(query_stats, conn, params=(username,))
    
    count = df_stats.iloc[0]['count']
    total = df_stats.iloc[0]['total']
    
    if pd.isna(total): total = 0
    avg = round(total / count, 1) if count > 0 else 0.0
    
    # Recent history (Added to maintain compatibility with app.py)
    c = conn.cursor()
    c.execute('''
        SELECT standard_code as subject, score, created_at 
        FROM quiz_history 
        WHERE username = ? 
        ORDER BY created_at DESC 
        LIMIT 5
    ''', (username,))
    recent_rows = c.fetchall()
    
    conn.close()
    
    return {
        "username": username,
        "solved_count": count,
        "total_score": round(total, 1),
        "avg_score": avg,
        "recent_history": recent_rows
    }

def update_progress(username, level, xp):
    """Update user level and xp."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET level = ?, xp = ? WHERE username = ?", (level, xp, username))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# [퀴즈 및 랭킹]
# ---------------------------------------------------------

def save_quiz_result(username, subject, score):
    """퀴즈 결과 저장"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO quiz_history (username, standard_code, score, created_at) VALUES (?, ?, ?, ?)",
              (username, subject, score, datetime.now()))
    conn.commit()
    conn.close()

def get_leaderboard_data():
    """랭킹 데이터 조회 (총점 기준 내림차순)"""
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT username as 사용자, 
               COUNT(*) as 풀이수, 
               SUM(score) as 총점,
               ROUND(AVG(score), 1) as 평균
        FROM quiz_history 
        GROUP BY username 
        ORDER BY 총점 DESC 
        LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    
    # 순위 컬럼 추가
    df.insert(0, '순위', range(1, 1 + len(df)))
    
    conn.close()
    return df

# ---------------------------------------------------------
# [오답 노트]
# ---------------------------------------------------------

def save_review_note(username, standard_code, question, answer, score):
    """오답 노트 저장"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO review_notes (username, standard_code, question, answer, score, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username, standard_code, question, answer, score, datetime.now()))
    conn.commit()
    conn.close()

def get_user_review_notes(username):
    """특정 사용자의 오답 노트 조회"""
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT * FROM review_notes WHERE username = ? ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def delete_review_note(note_id):
    """오답 노트 삭제 (복습 완료)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM review_notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()

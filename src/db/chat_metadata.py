import sqlite3
from datetime import datetime

from src.config import SQLITE_DB_FILE



def init_chat_metadata_db():
    """
    채팅 메타데이터 테이블이 없으면 생성
    """
    conn = sqlite3.connect(SQLITE_DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_metadata ( 
            thread_id TEXT PRIMARY KEY,
            chat_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()


def save_chat_metadata(thread_id: str, chat_name: str):
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO chat_metadata 
            (thread_id, chat_name, created_at, updated_at, message_count) 
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)
        ''', (thread_id, chat_name))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"채팅 메타데이터 저장 오류: {e}")
        return False
    

def update_chat_metadata(thread_id: str):
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE chat_metadata 
            SET message_count = message_count + 1, 
                updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
        ''', (thread_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"채팅 메타데이터 업데이트 오류: {e}")
        return False
    

def get_chat_list():
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        query = '''
            SELECT cm.thread_id, cm.chat_name, cm.created_at, cm.updated_at, cm.message_count
            FROM chat_metadata cm
            ORDER BY cm.updated_at DESC
        '''
        cursor.execute(query)
        chats = []
        
        for row in cursor.fetchall():
            thread_id, chat_name, created_at, updated_at, message_count = row
            try:
                dt = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
                formatted_date = dt.strftime('%m/%d %H:%M')
            except:
                formatted_date = "N/A"
            
            display_name = f"{chat_name} ({message_count}) - {formatted_date}"
            chats.append((display_name, thread_id))
        
        conn.close()
        return chats
        
    except Exception as e:
        print(f"채팅 목록 로드 오류: {e}")
        return []
    

def delete_chat(thread_id: str):
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        # 세 개의 테이블에서 모두 삭제
        cursor.execute('DELETE FROM writes WHERE thread_id = ?', (thread_id,))
        cursor.execute('DELETE FROM checkpoints WHERE thread_id = ?', (thread_id,))
        cursor.execute('DELETE FROM chat_metadata WHERE thread_id = ?', (thread_id,))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"채팅 삭제 오류: {e}")
        return False
    

def rename_chat(thread_id: str, new_name: str):
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE chat_metadata 
            SET chat_name = ?, updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
        ''', (new_name, thread_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"채팅 이름 변경 오류: {e}")
        return False
    

def generate_chat_name_from_message(message: str) -> str:
    if len(message) > 30:
        return message[:27] + "..."
    return message


def get_chat_name(thread_id: str) -> str:
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT chat_name FROM chat_metadata WHERE thread_id = ?', (thread_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "알 수 없는 채팅"
    except Exception as e:
        print(f"채팅 이름 조회 오류: {e}")
        return "오류"
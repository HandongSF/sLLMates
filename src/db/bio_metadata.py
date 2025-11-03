
import sqlite3
import uuid
from datetime import datetime
from typing import Optional, List, Dict
from src.db.vector_store import BioChromaDBVectorStore

from src.config import BIO_SQLITE_DB_FILE


# ==================== SQLite DB 연결 관리 ====================

def get_bio_db_connection():
    """Bio SQLite DB 연결을 반환합니다."""
    conn = sqlite3.connect(BIO_SQLITE_DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def close_bio_db_connection(conn):
    """Bio SQLite DB 연결을 닫습니다."""
    if conn:
        conn.close()


def init_bio_db():
    """Bio SQLite DB를 초기화합니다."""
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bio_memory (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 3,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    close_bio_db_connection(conn)
    print(f"[Bio DB] SQLite DB 초기화 완료: {BIO_SQLITE_DB_FILE}")


# ==================== ChromaDB 연결 헬퍼 ====================

def get_bio_chroma_collection():
    """
    Bio ChromaDB Collection을 반환합니다.
    각 함수 호출마다 독립적으로 연결을 생성합니다.
    
    Returns:
        ChromaDB Collection 객체
    """
    try:
        vector_store = BioChromaDBVectorStore()
        collection = vector_store.get_collection()
        return collection
    except Exception as e:
        print(f"[Bio DB] ChromaDB 연결 실패: {e}")
        return None

# ==================== CRUD 함수 (자동으로 Collection 관리) ====================

def add_bio_with_vector(bio_id: str, text: str, 
                       importance: int, vector: List[float]) -> str:
    """
    새로운 bio를 SQLite DB와 Vector Store에 추가합니다.
    ChromaDB 연결을 자동으로 생성하고 정리합니다.
    
    Args:
        bio_id: Bio ID
        text: Bio 텍스트
        importance: 중요도 (1-10)
        vector: 임베딩 벡터
    
    Returns:
        생성된 bio의 ID
    """
    now = datetime.now().isoformat()
    
    # SQLite에 저장
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO bio_memory (id, text, importance, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (bio_id, text, importance, now, now))
        
        conn.commit()
        print(f"[Bio DB] SQLite 저장 완료 (ID: {bio_id[:8]}...)")
        
        # Vector Store에 저장
        collection = get_bio_chroma_collection()
        if collection:
            try:
                collection.add(
                    ids=[bio_id],
                    embeddings=[vector],
                    documents=[text],
                    metadatas=[{
                        "id": bio_id,
                        "importance": importance,
                        "last_updated": now
                    }]
                )
                print(f"[Bio DB] Vector Store 저장 완료 (ID: {bio_id[:8]}...)")
            except Exception as e:
                print(f"[Bio DB] Vector Store 저장 실패: {e}")
        
        return bio_id
        
    except Exception as e:
        conn.rollback()
        print(f"[Bio DB] Bio 추가 실패: {e}")
        raise
    finally:
        close_bio_db_connection(conn)


def update_bio_with_vector(bio_id: str, text: Optional[str] = None,
                           importance: Optional[int] = None, 
                           vector: Optional[List[float]] = None):
    """
    기존 bio를 업데이트합니다.
    ChromaDB 연결을 자동으로 생성하고 정리합니다.
    
    Args:
        bio_id: 업데이트할 bio의 ID
        text: 새로운 텍스트 (None이면 유지)
        importance: 새로운 중요도 (None이면 유지)
        vector: 새로운 임베딩 벡터 (text 변경 시 필수)
    """
    now = datetime.now().isoformat()
    
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    try:
        # 기존 데이터 조회
        cursor.execute("SELECT text, importance FROM bio_memory WHERE id = ?", (bio_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
            return
        
        current_text, current_importance = result
        new_text = text if text is not None else current_text
        new_importance = importance if importance is not None else current_importance
        
        # SQLite 업데이트
        cursor.execute("""
            UPDATE bio_memory 
            SET text = ?, importance = ?, updated_at = ?
            WHERE id = ?
        """, (new_text, new_importance, now, bio_id))
        
        conn.commit()
        print(f"[Bio DB] SQLite 업데이트 완료 (ID: {bio_id[:8]}...)")
        
        # Vector Store 업데이트
        collection = get_bio_chroma_collection()
        if collection and vector:
            try:
                collection.update(
                    ids=[bio_id],
                    embeddings=[vector],
                    documents=[new_text],
                    metadatas=[{
                        "id": bio_id,
                        "importance": new_importance,
                        "last_updated": now
                    }]
                )
                print(f"[Bio DB] Vector Store 업데이트 완료 (ID: {bio_id[:8]}...)")
            except Exception as e:
                print(f"[Bio DB] Vector Store 업데이트 실패: {e}")
        
    except Exception as e:
        conn.rollback()
        print(f"[Bio DB] Bio 업데이트 실패: {e}")
        raise
    finally:
        close_bio_db_connection(conn)

def add_bio(text: str, importance: int = 3, bio_id: Optional[str] = None) -> str:
    """
    새로운 bio 문장을 SQLite DB와 Vector Store에 추가합니다.
    
    Args:
        text: bio 문장
        importance: 중요도 (1-10)
        bio_id: 지정할 ID (없으면 자동 생성)
    
    Returns:
        생성된 bio의 ID
    """
    if bio_id is None:
        bio_id = str(uuid.uuid4())
    
    now = datetime.now().isoformat()
    
    # SQLite에 저장
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO bio_memory (id, text, importance, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (bio_id, text, importance, now, now))
    
    conn.commit()
    close_bio_db_connection(conn)
    
    # Vector Store 저장
    collection = get_bio_chroma_collection()
    if collection:
        try:
            collection.add(
                ids=[bio_id],
                documents=[text],
                metadatas=[{"importance": importance, "last_updated": now}]
            )
            print(f"[Bio DB] 새로운 bio 추가 완료 (ID: {bio_id[:8]}...): {text[:50]}...")
        except Exception as e:
            print(f"[Bio DB] Vector Store 저장 실패: {e}")
    
    return bio_id


def update_bio(bio_id: str, text: Optional[str] = None, importance: Optional[int] = None):
    """
    기존 bio 문장을 업데이트합니다.
    
    Args:
        bio_id: 업데이트할 bio의 ID
        text: 새로운 텍스트 (None이면 유지)
        importance: 새로운 중요도 (None이면 유지)
    """
    now = datetime.now().isoformat()
    
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    # 기존 데이터 조회
    cursor.execute("SELECT text, importance FROM bio_memory WHERE id = ?", (bio_id,))
    result = cursor.fetchone()
    
    if not result:
        close_bio_db_connection(conn)
        print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
        return
    
    current_text, current_importance = result
    new_text = text if text is not None else current_text
    new_importance = importance if importance is not None else current_importance
    
    # SQLite 업데이트
    cursor.execute("""
        UPDATE bio_memory 
        SET text = ?, importance = ?, updated_at = ?
        WHERE id = ?
    """, (new_text, new_importance, now, bio_id))
    
    conn.commit()
    close_bio_db_connection(conn)
    
    # Vector Store 업데이트
    collection = get_bio_chroma_collection()
    if collection:
        try:
            collection.update(
                ids=[bio_id],
                documents=[new_text],
                metadatas=[{"importance": new_importance, "last_updated": now}]
            )
            print(f"[Bio DB] Bio 업데이트 완료 (ID: {bio_id[:8]}...)")
        except Exception as e:
            print(f"[Bio DB] Vector Store 업데이트 실패: {e}")


def delete_bio(bio_id: str):
    """
    bio를 SQLite DB와 Vector Store에서 삭제합니다.
    
    Args:
        bio_id: 삭제할 bio의 ID
    """
    # SQLite에서 삭제
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM bio_memory WHERE id = ?", (bio_id,))
    deleted_count = cursor.rowcount
    
    conn.commit()
    close_bio_db_connection(conn)
    
    if deleted_count == 0:
        print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
        return
    
    # Vector Store에서 삭제
    collection = get_bio_chroma_collection()
    if collection:
        try:
            collection.delete(ids=[bio_id])
            print(f"[Bio DB] Bio 삭제 완료 (ID: {bio_id[:8]}...)")
        except Exception as e:
            print(f"[Bio DB] Vector Store 삭제 실패: {e}")


def get_all_bios() -> List[Dict]:
    """
    모든 bio 데이터를 조회합니다.
    
    Returns:
        bio 데이터 리스트 [{"id": str, "text": str, "importance": int, ...}, ...]
    """
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, text, importance, created_at, updated_at 
        FROM bio_memory 
        ORDER BY updated_at DESC
    """)
    
    results = cursor.fetchall()
    close_bio_db_connection(conn)
    
    bios = []
    for row in results:
        bios.append({
            "id": row[0],
            "text": row[1],
            "importance": row[2],
            "created_at": row[3],
            "updated_at": row[4]
        })
    
    return bios


def get_bio_by_id(bio_id: str) -> Optional[Dict]:
    """
    특정 ID의 bio를 조회합니다.
    
    Args:
        bio_id: 조회할 bio의 ID
    
    Returns:
        bio 데이터 딕셔너리 또는 None
    """
    conn = get_bio_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, text, importance, created_at, updated_at 
        FROM bio_memory 
        WHERE id = ?
    """, (bio_id,))
    
    result = cursor.fetchone()
    close_bio_db_connection(conn)
    
    if not result:
        return None
    
    return {
        "id": result[0],
        "text": result[1],
        "importance": result[2],
        "created_at": result[3],
        "updated_at": result[4]
    }


def save_or_update_bio(new_bio_blocks: List[Dict], similarity_threshold: float = 0.85):
    """
    새로운 bio 문장들을 저장하거나 기존 문장을 업데이트합니다.
    유사도가 높은 기존 문장이 있으면 업데이트하고, 없으면 새로 추가합니다.
    
    Args:
        new_bio_blocks: [{"text": "...", "importance": int}, ...]
        
        similarity_threshold: 유사도 임계값 (기본 0.85)
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] Vector Store가 초기화되지 않았습니다.")
        return
    
    for bio in new_bio_blocks:
        text = bio["text"].strip()
        importance = bio.get("importance", 3)
        
        if not text:
            continue
        
        try:
            results = collection.query(query_texts=[text], n_results=3)
            is_updated = False
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, distance in enumerate(results['distances'][0]):
                    similarity = 1 - distance / 2
                    
                    if similarity > similarity_threshold:
                        bio_id = results['ids'][0][i]
                        print(f"[Bio DB] 기존 항목 갱신됨 (similarity={similarity:.2f}): {text[:50]}...")
                        update_bio(bio_id, text=text, importance=importance)
                        is_updated = True
                        break
            
            if not is_updated:
                add_bio(text, importance)
                
        except Exception as e:
            print(f"[Bio DB] 유사도 검색 실패: {e}, 새로운 bio로 추가합니다.")
            add_bio(text, importance)
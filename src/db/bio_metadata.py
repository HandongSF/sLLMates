import uuid
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
from src.db.vector_store import BioChromaDBVectorStore


# ==================== ChromaDB 연결 헬퍼 ====================

_vector_store_instance = None
_vector_store_collection = None

def get_bio_chroma_collection():
    """
    미리 초기화된 싱글톤 ChromaDB Collection을 반환합니다.
    """
    if _vector_store_collection is None:
        print("[Bio DB] ChromaDB가 초기화되지 않았습니다. init_bio_db()를 호출합니다.")
        # 혹은 여기서 자동으로 init_bio_db()를 호출하게 할 수도 있습니다.
        return init_bio_db()
    return _vector_store_collection



def init_bio_db():
    global _vector_store_instance, _vector_store_collection
    if _vector_store_instance is None:
        try:
            # 🌟 여기서 CUDA OOM이 발생할 수 있습니다 (정상)
            _vector_store_instance = BioChromaDBVectorStore() 
            _vector_store_collection = _vector_store_instance.get_bio_collection()
            
            if _vector_store_collection:
                print(f"[Bio DB] ChromaDB 초기화 및 싱글톤 생성 완료: {_vector_store_collection.name}")
            else:
                print("[Bio DB] ChromaDB 컬렉션 가져오기 실패")
                
        except Exception as e:
            print(f"[Bio DB] ChromaDB 초기화 중 치명적 오류: {e}")
            # 여기서 오류를 다시 발생시켜 앱을 중지시키거나 적절히 처리
            raise e
    return _vector_store_collection


# ==================== CRUD 함수 ====================

def add_bio(text: str, importance: int = 3, bio_id: Optional[str] = None) -> str:
    """
    새로운 bio 문장을 ChromaDB에 추가합니다.
    
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
    
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return bio_id
    
    try:
        collection.add(
            ids=[bio_id],
            documents=[text],
            metadatas=[{
                "importance": importance,
                "last_updated": now
            }]
        )
        print(f"[Bio DB] 새로운 bio 추가 완료 (ID: {bio_id[:8]}...): {text[:50]}...")
        return bio_id
        
    except Exception as e:
        print(f"[Bio DB] Bio 추가 실패: {e}")
        raise


def add_bio_with_vector(bio_id: str, text: str, 
                       importance: int, vector: List[float]) -> str:
    """
    새로운 bio를 사전 계산된 벡터와 함께 ChromaDB에 추가합니다.
    
    Args:
        bio_id: Bio ID
        text: Bio 텍스트
        importance: 중요도 (1-10)
        vector: 임베딩 벡터
    
    Returns:
        생성된 bio의 ID
    """
    now = datetime.now().isoformat()
    
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return bio_id
    
    try:
        collection.add(
            ids=[bio_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{
                "importance": importance,
                "last_updated": now
            }]
        )
        print(f"[Bio DB] Bio 추가 완료 (ID: {bio_id[:8]}...)")
        return bio_id
        
    except Exception as e:
        print(f"[Bio DB] Bio 추가 실패: {e}")
        raise


def update_bio(bio_id: str, text: Optional[str] = None, 
               importance: Optional[int] = None):
    """
    기존 bio 문장을 업데이트합니다.
    
    Args:
        bio_id: 업데이트할 bio의 ID
        text: 새로운 텍스트 (None이면 유지)
        importance: 새로운 중요도 (None이면 유지)
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return
    
    try:
        # 기존 데이터 조회 (단일 ID만 조회하므로 안전)
        existing = collection.get(ids=[bio_id])
        
        if not existing['ids']:
            print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
            return
        
        current_metadata = existing['metadatas'][0]
        current_text = existing['documents'][0]
        
        new_text = text if text is not None else current_text
        new_importance = importance if importance is not None else current_metadata.get('importance', 3)
        now = datetime.now().isoformat()
        
        # ChromaDB 업데이트
        collection.update(
            ids=[bio_id],
            documents=[new_text],
            metadatas=[{
                "importance": new_importance,
                "last_updated": now
            }]
        )
        print(f"[Bio DB] Bio 업데이트 완료 (ID: {bio_id[:8]}...)")
        
    except Exception as e:
        print(f"[Bio DB] Bio 업데이트 실패: {e}")
        raise


def update_bio_with_vector(bio_id: str, text: Optional[str] = None,
                           importance: Optional[int] = None, 
                           vector: Optional[List[float]] = None):
    """
    기존 bio를 사전 계산된 벡터와 함께 업데이트합니다.
    
    Args:
        bio_id: 업데이트할 bio의 ID
        text: 새로운 텍스트 (None이면 유지)
        importance: 새로운 중요도 (None이면 유지)
        vector: 새로운 임베딩 벡터 (text 변경 시 필수)
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return
    
    try:
        # 기존 데이터 조회 (단일 ID만 조회하므로 안전)
        existing = collection.get(ids=[bio_id])
        
        if not existing['ids']:
            print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
            return
        
        current_metadata = existing['metadatas'][0]
        current_text = existing['documents'][0]
        
        new_text = text if text is not None else current_text
        new_importance = importance if importance is not None else current_metadata.get('importance', 3)
        now = datetime.now().isoformat()
        
        # ChromaDB 업데이트
        update_params = {
            "ids": [bio_id],
            "documents": [new_text],
            "metadatas": [{
                "importance": new_importance,
                "last_updated": now
            }]
        }
        
        if vector is not None:
            update_params["embeddings"] = [vector]
        
        collection.update(**update_params)
        print(f"[Bio DB] Bio 업데이트 완료 (ID: {bio_id[:8]}...)")
        
    except Exception as e:
        print(f"[Bio DB] Bio 업데이트 실패: {e}")
        raise


def delete_bio(bio_id: str):
    """
    bio를 ChromaDB에서 삭제합니다.
    
    Args:
        bio_id: 삭제할 bio의 ID
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return
    
    try:
        # 존재 여부 확인 (단일 ID만 조회하므로 안전)
        existing = collection.get(ids=[bio_id])
        if not existing['ids']:
            print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
            return
        
        # ChromaDB에서 삭제
        collection.delete(ids=[bio_id])
        print(f"[Bio DB] Bio 삭제 완료 (ID: {bio_id[:8]}...)")
        
    except Exception as e:
        print(f"[Bio DB] Bio 삭제 실패: {e}")


def get_all_bios() -> List[Dict]:
    """
    모든 bio 데이터를 페이징 방식으로 조회합니다. (API 사용)
    'embeddings' (벡터)는 제외하고 가져와 메모리 문제를 방지합니다.
    
    Returns:
        bio 데이터 리스트 [{"id": str, "document": str, "importance": int, "last_updated": str}, ...]
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return []
    
    try:
        # 🌟 핵심: include를 명시하여 'embeddings'를 제외합니다.
        results = collection.get(
            # documents와 metadatas만 요청합니다.
            include=["documents", "metadatas"] 
        )
        
        bios = []
        if results['ids']:
            for i in range(len(results['ids'])):
                bio_id = results['ids'][i]
                doc = results['documents'][i]
                meta = results['metadatas'][i]
                
                bios.append({
                    "id": bio_id,
                    "document": doc,
                    "importance": meta.get('importance', 3),
                    "last_updated": meta.get('last_updated', '')
                })
        
        # 참고: get() API는 정렬을 보장하지 않을 수 있습니다. 
        # SQL의 ORDER BY가 꼭 필요했다면, 여기에서 Python으로 정렬해야 합니다.
        if bios:
            bios.sort(key=lambda x: x['last_updated'], reverse=True)
            
        return bios
        
    except Exception as e:
        print(f"[Bio DB] Bio 조회 실패: {e}")
        return []


def get_bio_by_id(bio_id: str) -> Optional[Dict]:
    """
    특정 ID의 bio를 조회합니다. (API 사용)
    벡터는 제외하고 가져옵니다.
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return None
    
    try:
        result = collection.get(
            ids=[bio_id],
            # documents와 metadatas만 요청합니다.
            include=["documents", "metadatas"]
        )
        
        if not result['ids']:
            return None
        
        doc = result['documents'][0]
        meta = result['metadatas'][0]
        
        return {
            "id": bio_id,
            "document": doc,
            "importance": meta.get('importance', 3),
            "last_updated": meta.get('last_updated', '')
        }
        
    except Exception as e:
        print(f"[Bio DB] Bio 조회 실패: {e}")
        return None

def count_all_bios() -> int:
    """
    전체 bio 개수를 반환합니다. (API 사용)
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return 0
    
    try:
        count = collection.count()
        return count
    except Exception as e:
        print(f"[Bio DB] Bio 개수 조회 실패: {e}")
        return 0

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
        print("[Bio DB] ChromaDB가 초기화되지 않았습니다.")
        return
    
    for bio in new_bio_blocks:
        text = bio["text"].strip()
        importance = bio.get("importance", 3)
        
        if not text:
            continue
        
        try:
            # 유사한 기존 bio 검색
            results = collection.query(query_texts=[text], n_results=3)
            is_updated = False
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, distance in enumerate(results['distances'][0]):
                    # ChromaDB는 L2 distance를 반환하므로 similarity로 변환
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


def search_similar_bios(query_text: str, n_results: int = 5) -> List[Dict]:
    """
    쿼리 텍스트와 유사한 bio들을 검색합니다.
    (수정: ChromaDB 반환 형식에 맞게 수정됨)
    """
    collection = get_bio_chroma_collection()
    if collection is None:
        print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
        return []
    
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        
        similar_bios = []
        
        # --- 🌟 수정된 부분 🌟 ---
        # query()는 쿼리 리스트의 각 항목에 대한 결과 리스트를 반환합니다. (이중 리스트)
        # 우리는 쿼리를 하나만 보냈으므로(query_texts=[query_text]),
        # 항상 첫 번째 결과 리스트(예: results['ids'][0])를 사용해야 합니다.
        
        # results['ids']가 비어있지 않고, '그 첫 번째 결과 리스트'도 비어있는지 확인
        if results['ids'] and len(results['ids'][0]) > 0:
            
            # 첫 번째 쿼리(인덱스 0)의 결과 리스트들을 가져옵니다.
            ids_list = results['ids'][0]
            docs_list = results['documents'][0]
            metadatas_list = results['metadatas'][0]
            distances_list = results['distances'][0]
            
            # 디버그 프린트: 여기서 실제 검색된 항목의 개수를 확인
            print("HeRE:: " + str(len(ids_list))) 

            # 이제 '항목의 개수'만큼 순회합니다.
            for i in range(len(ids_list)):
                document = docs_list[i]     # [i] 사용
                metadata = metadatas_list[i] # [i] 사용
                distance = distances_list[i] # [i] 사용
                
                similarity = 1 - distance / 2
                score = metadata.get('importance', 3) * similarity
                
                similar_bios.append({
                    "document": document,
                    "score": score
                })
        
        # (이하 로직은 동일)
        similar_bios.sort(key=lambda x: x['score'], reverse=True)
        
        return similar_bios
        
    except Exception as e:
        print(f"[Bio DB] 유사 bio 검색 실패: {e}")
        return []
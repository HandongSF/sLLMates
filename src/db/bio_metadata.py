import uuid
from datetime import datetime
from typing import Optional, List, Dict

class BioMetadata:
    
    def __init__(self, bio_chroma_db):
        self.collection = bio_chroma_db.get_bio_vector_store()
        self.embedding_function = self.collection._embedding_function

    def get_bio_chroma_collection(self):
        return self.collection


    def add_bio(self, text: str, importance: int = 3, bio_id: Optional[str] = None) -> str:
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
        
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
            return bio_id
        
        vector = self.embedding_function.embed_query(text)
        
        try:
            collection._collection.add(
                ids=[bio_id],
                documents=[text],
                embeddings=[vector],
                metadatas=[{
                    "bio_id": bio_id,
                    "importance": importance,
                    "last_updated": now
                }]
            )
            print(f"[Bio DB] 새로운 bio 추가 완료 (ID: {bio_id[:8]}...): {text[:50]}...")
            return bio_id
            
        except Exception as e:
            print(f"[Bio DB] Bio 추가 실패: {e}")
            raise


    def update_bio(self, bio_id: str, text: Optional[str] = None, 
                importance: Optional[int] = None):
        """
        기존 bio 문장을 업데이트합니다.
        
        Args:
            bio_id: 업데이트할 bio의 ID
            text: 새로운 텍스트 (None이면 유지)
            importance: 새로운 중요도 (None이면 유지)
        """
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
            return
        
        try:
            # 기존 데이터 조회 (단일 ID만 조회하므로 안전)
            existing = collection._collection.get(ids=[bio_id])
            
            if not existing['ids']:
                print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
                return
            
            current_metadata = existing['metadatas'][0]
            current_text = existing['documents'][0]
            
            new_text = text if text is not None else current_text
            vector = self.embedding_function.embed_query(new_text)
            new_importance = importance if importance is not None else current_metadata.get('importance', 3)
            now = datetime.now().isoformat()
            
            # ChromaDB 업데이트
            collection._collection.update(
                ids=[bio_id],
                documents=[new_text],
                embeddings=[vector],
                metadatas=[{
                    "bio_id": bio_id,
                    "importance": new_importance,
                    "last_updated": now
                }]
            )
            print(f"[Bio DB] Bio 업데이트 완료 (ID: {bio_id[:8]}...)")
            
        except Exception as e:
            print(f"[Bio DB] Bio 업데이트 실패: {e}")
            raise


    def delete_bio(self, bio_id: str):
        """
        bio를 ChromaDB에서 삭제합니다.
        
        Args:
            bio_id: 삭제할 bio의 ID
        """
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
            return
        
        try:
            # 존재 여부 확인 (단일 ID만 조회하므로 안전)
            existing = collection._collection.get(ids=[bio_id])
            if not existing['ids']:
                print(f"[Bio DB] ID {bio_id}를 찾을 수 없습니다.")
                return
            
            # ChromaDB에서 삭제
            collection._collection.delete(ids=[bio_id])
            print(f"[Bio DB] Bio 삭제 완료 (ID: {bio_id[:8]}...)")
            
        except Exception as e:
            print(f"[Bio DB] Bio 삭제 실패: {e}")


    def get_all_bios(self) -> List[Dict]:
        """
        모든 bio 데이터를 페이징 방식으로 조회합니다. (API 사용)
        'embeddings' (벡터)는 제외하고 가져와 메모리 문제를 방지합니다.
        
        Returns:
            bio 데이터 리스트 [{"id": str, "document": str, "importance": int, "last_updated": str}, ...]
        """
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
            return []
        
        try:
            #include를 명시하여 'embeddings'를 제외합니다.
            results = collection._collection.get(
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


    def get_bio_by_id(self, bio_id: str) -> Optional[Dict]:
        """
        특정 ID의 bio를 조회합니다. (API 사용)
        벡터는 제외하고 가져옵니다.
        """
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
            return None
        
        try:
            result = collection._collection.get(
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


    def count_all_bios(self) -> int:
        """
        전체 bio 개수를 반환합니다. (API 사용)
        """
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
            return 0
        
        try:
            count = collection._collection.count()
            return count
        except Exception as e:
            print(f"[Bio DB] Bio 개수 조회 실패: {e}")
            return 0


    def save_or_update_bio(self, new_bio_blocks: List[Dict], distance_threshold: float = 0.25):
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB가 초기화되지 않았습니다.")
            return
        
        for block in new_bio_blocks:
            text = block["content"].strip()
            importance = block.get("importance", 3)
            
            if not text:
                continue
            
            try:
                # 유사한 기존 bio 검색
                results = collection.similarity_search_with_score(text, k=3)
                is_updated = False
                
                for doc, score in results:
                    if score < distance_threshold:
                        bio_id = doc.metadata.get("bio_id")

                        print(f"[Bio DB] 기존 항목 갱신됨 (distance={score:.2f}): {text[:50]}...")

                        self.update_bio(
                            bio_id,
                            text=text,
                            importance=importance
                        )

                        is_updated = True
                        break

                if not is_updated:
                    self.add_bio(text, importance)
                    
            except Exception as e:
                print(f"[Bio DB] 유사도 검색 실패: {e}, 새로운 bio로 추가합니다.")
                self.add_bio(text, importance)


    def search_similar_bios(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        쿼리 텍스트와 유사한 bio들을 검색
        """
        collection = self.get_bio_chroma_collection()
        if collection is None:
            print("[Bio DB] ChromaDB에 연결할 수 없습니다.")
            return []
        
        try:
            results = collection.similarity_search_with_score(
            query_text,
            k=n_results
            )
            
            similar_bios = []
            
            for doc, score in results:
                similarity = 1 - score
                importance = doc.metadata.get("importance", 3)
                final_score = importance * similarity
                similar_bios.append({
                    "document": doc.page_content,
                    "score": final_score
                })

            similar_bios.sort(key=lambda x: x["score"], reverse=True)

            return similar_bios

        except Exception as e:
            print(f"[Bio DB] 유사 bio 검색 실패: {e}")
            return []
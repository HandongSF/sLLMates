"""
BioManager - Bio 데이터 임베딩 및 저장을 위한 멀티프로세싱 관리 클래스
"""
import os
import re
from multiprocessing import Process, Queue
from typing import List, Dict
import time
import uuid

from src.db.bio_metadata import add_bio_with_vector, update_bio_with_vector
from src.db.vector_store import BioChromaDBVectorStore


class BioManager:
    """
    Bio 데이터를 임베딩하고 저장하는 멀티프로세스 관리 클래스
    
    Architecture:
    1. bio_processor: 메인에서 호출하는 진입점 (비동기, 즉시 반환)
    2. bio_worker: bio_sentences 리스트 전체를 임베딩하여 Queue에 추가
    3. bio_writer: 백그라운드에서 Queue의 데이터를 Vector Store에 저장
    """
    
    def __init__(self):
        """BioManager 초기화"""
        self.embedding_queue = Queue(maxsize=100)  # 임베딩된 데이터를 담는 큐
        self.writer_process = None
        
        # Vector Store 초기화 (메인 프로세스에서)
        #self.vector_store = BioChromaDBVectorStore()
        
        # 백그라운드 Writer 프로세스 시작
        #self._start_writer_process()
        
        print(f"[BioManager] 초기화 완료: Writer 프로세스 실행 중")


    def extract_bio_with_importance(self, text: str) -> List[Dict[str, any]]:
        pattern = r"<bio>(.*?)</bio>\s*<importance>\s*(\d+)\s*</importance>"
        results = re.findall(pattern, text, re.DOTALL)
        
        bio_list = []
        for bio_text, importance in results:
            cleaned = bio_text.strip()
            if not cleaned:
                continue
            importance_value = max(1, min(int(importance), 10))
            bio_list.append({
                "text": cleaned,
                "importance": importance_value
            })
        return bio_list
    
    def clean_bio_tags(self, text: str) -> str:
        return re.sub(
            r"<bio>.*?</bio>", 
            "", 
            text, 
            flags=re.DOTALL
        ).strip()
    
    def _start_writer_process(self):
        """백그라운드 Writer 프로세스 시작"""
        self.writer_process = Process(
            target=self._bio_writer,
            args=(self.embedding_queue,),
            daemon=True
        )
        self.writer_process.start()
        print(f"[BioManager] Writer 프로세스 시작 (PID: {self.writer_process.pid})")
    
    @staticmethod
    def _bio_writer(queue: Queue):
        """
        백그라운드에서 상시 실행되는 Writer 프로세스
        Queue에서 임베딩된 데이터를 꺼내서 Vector Store에 저장
        
        Args:
            queue: 임베딩된 bio 데이터가 담긴 Queue
        """
        print(f"[Bio Writer] 시작 (PID: {os.getpid()})")
        
        while True:
            try:
                # Queue에서 데이터 가져오기 (timeout=1초)
                data = queue.get(timeout=1)
                
                if data is None:  # 종료 신호
                    print("[Bio Writer] 종료 신호 수신")
                    break
                
                # 데이터 저장
                bio_id = data.get("bio_id")
                text = data.get("text")
                importance = data.get("importance")
                vector = data.get("vector")
                is_update = data.get("is_update", False)
                
                if is_update:
                    # 업데이트
                    update_bio_with_vector(
                        bio_id=bio_id,
                        text=text,
                        importance=importance,
                        vector=vector
                    )
                    print(f"[Bio Writer] Bio 업데이트 완료 (ID: {bio_id[:8]}...)")
                else:
                    # 새로 추가
                    add_bio_with_vector(
                        bio_id=bio_id,
                        text=text,
                        importance=importance,
                        vector=vector
                    )
                    print(f"[Bio Writer] Bio 추가 완료 (ID: {bio_id[:8]}...)")
                
            except Exception as e:
                if "Empty" not in str(type(e).__name__):  # Queue Empty는 정상
                    print(f"[Bio Writer] 오류: {e}")
                time.sleep(0.1)  # CPU 사용률 감소
    
    @staticmethod
    def _bio_worker(bio_sentences: List[Dict], queue: Queue):
        """
        bio_sentences 리스트 전체를 임베딩하여 Queue에 추가하는 Worker
        
        Args:
            bio_sentences: [{"text": str, "importance": int, "bio_id": str (optional)}, ...]
            queue: 임베딩 결과를 담을 Queue
        """
        print(f"[Bio Worker] 시작 (PID: {os.getpid()}): {len(bio_sentences)}개 항목 처리 중...")
        
        try:
            # Vector Store 초기화
            vector_store = BioChromaDBVectorStore()
            
            for bio_item in bio_sentences:
                try:
                    text = bio_item.get("text", "").strip()
                    importance = bio_item.get("importance", 3)
                    bio_id = bio_item.get("bio_id")
                    is_update = bio_item.get("is_update", False)
                    
                    if not text:
                        continue
                    
                    # 임베딩 생성
                    vector = vector_store.embed_text(text)
                    
                    if vector is None:
                        print(f"[Bio Worker] 임베딩 실패: {text[:50]}...")
                        continue
                    
                    # Bio ID 생성 (없으면)
                    if not bio_id:
                        bio_id = str(uuid.uuid4())
                    
                    # Queue에 추가
                    queue.put({
                        "bio_id": bio_id,
                        "text": text,
                        "importance": importance,
                        "vector": vector,
                        "is_update": is_update
                    })
                    
                    print(f"[Bio Worker] 임베딩 완료: {text[:30]}...")
                    
                except Exception as e:
                    print(f"[Bio Worker] Bio 항목 처리 오류: {e}")
                    continue
            
            print(f"[Bio Worker] 완료 (PID: {os.getpid()}): {len(bio_sentences)}개 항목 처리 완료")
            
        except Exception as e:
            print(f"[Bio Worker] 전체 처리 오류: {e}")
    
    def bio_processor(self, bio_sentences: List[Dict]):
        """
        메인에서 호출하는 진입점
        bio_sentences 리스트 전체를 하나의 Worker 프로세스에 전달하고 즉시 반환
        
        Args:
            bio_sentences: [{"text": str, "importance": int}, ...]
        """
        if not bio_sentences:
            print("[BioManager] 처리할 Bio 데이터 없음")
            return
        
        print(f"[BioManager] Bio 처리 시작: {len(bio_sentences)}개 항목")
        
        try:
            # 새로운 Worker 프로세스 생성 (비동기)
            worker_process = Process(
                target=self._bio_worker,
                args=(bio_sentences, self.embedding_queue),
                daemon=True
            )
            worker_process.start()
            
            print(f"[BioManager] Worker 프로세스 시작 (PID: {worker_process.pid}), 즉시 반환")
            # 프로세스 완료를 기다리지 않고 즉시 반환
            
        except Exception as e:
            print(f"[BioManager] Bio 처리 오류: {e}")
    
    def shutdown(self):
        """BioManager 종료 및 리소스 정리"""
        print("[BioManager] 종료 시작...")
        
        # Writer 프로세스 종료
        if self.writer_process and self.writer_process.is_alive():
            self.embedding_queue.put(None)  # 종료 신호
            self.writer_process.join(timeout=5)
            
            if self.writer_process.is_alive():
                self.writer_process.terminate()
            
            print("[BioManager] Writer 프로세스 종료 완료")
        
        print("[BioManager] 모든 프로세스 종료 완료")
    
    def __del__(self):
        """소멸자: 자동 정리"""
        try:
            self.shutdown()
        except:
            pass
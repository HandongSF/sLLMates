"""
Bio (Personalized Memory) Evaluation Runner

실험 구조:
  Round 1: Bio 추출 → ChromaDB 저장 (SET별 컬렉션)
  Round 2: precision_tasks 평가 (Yes/No 답변, 자동 채점)
  Round 3: implicit_generation_tasks 평가 (수동 채점용 CSV)
"""

import sys
import os
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
# 이 스크립트는 프로젝트 루트에서 실행하거나, sys.path에 루트를 추가해서 사용하세요.
# 예) python bio_eval/bio_eval_runner.py --dataset path/to/dataset.json

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from llama_cpp import Llama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import convert_to_openai_messages
from langchain.schema import BaseMessage
import importlib

from src.config import SELECTED_CONFIG_FILE, MODELS_DIR
from src.core.parsers import parse_llm_output, parse_bio_with_importance

# ── 평가 전용 ChromaDB / Embedding ────────────────────────────────────────────
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings

# ══════════════════════════════════════════════════════════════════════════════
# 설정 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_config():
    module_name = f"src.configs.{SELECTED_CONFIG_FILE}"
    if module_name.endswith('.py'):
            module_name = module_name[:-3]
    module = importlib.import_module(module_name)
    return getattr(module, "CONFIG", {})


# ══════════════════════════════════════════════════════════════════════════════
# LLM 초기화
# ══════════════════════════════════════════════════════════════════════════════

def init_llm(config: Dict) -> Llama:
    cfg = config["CHAT_MODEL_CONFIG"]
    llm = Llama(
        model_path=cfg.get("model_path", ""),
        n_gpu_layers=cfg.get("n_gpu_layers", 0),
        main_gpu=cfg.get("main_gpu", 0),
        tensor_split=cfg.get("tensor_split", None),
        use_mmap=cfg.get("use_mmap", True),
        use_mlock=cfg.get("use_mlock", False),
        n_ctx=cfg.get("n_ctx", 4096),
        n_batch=cfg.get("n_batch", 512),
        flash_attn=cfg.get("flash_attn", False),
        verbose=cfg.get("verbose", False),
    )
    return llm


# ══════════════════════════════════════════════════════════════════════════════
# Embedding 초기화
# ══════════════════════════════════════════════════════════════════════════════

def init_embedding(config: Dict) -> HuggingFaceEmbeddings:
    emb_cfg = config["EMBEDDING_MODEL_CONFIG"]
    return HuggingFaceEmbeddings(
        model_name=emb_cfg.get("model_name", ""),
        model_kwargs=emb_cfg.get("model_kwargs", {}),
        encode_kwargs=emb_cfg.get("encode_kwargs", {}),
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM 호출 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(llm: Llama, config: Dict, messages: List[Dict], formatter=None) -> str:
    cfg = config["CHAT_MODEL_CONFIG"]
    common_kwargs = dict(
        max_tokens=cfg.get("max_tokens", -1),
        temperature=cfg.get("temperature", 0.6),
        top_p=cfg.get("top_p", 0.95),
        min_p=cfg.get("min_p", 0.0),
        stop=cfg.get("stop", []),
        top_k=cfg.get("top_k", 20),
    )

    if formatter:
        full_prompt = formatter(messages=messages).prompt
        response_data = llm.create_completion(prompt=full_prompt, **common_kwargs)
        return response_data["choices"][0]["text"].strip()
    else:
        response_data = llm.create_chat_completion(messages=messages, **common_kwargs)
        return response_data["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════════════════════
# Formatter 초기화
# ══════════════════════════════════════════════════════════════════════════════

def init_formatter(config: Dict):
    if not config.get("USE_CUSTOM_CHAT_HANDLER", False):
        return None
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        fc = config.get("FORMATTER_CONFIG", {})
        tmpl = config.get("CUSTOM_CHAT_TEMPLATE", "")
        if tmpl and fc.get("eos_token") and fc.get("bos_token"):
            return Jinja2ChatFormatter(
                template=tmpl,
                eos_token=fc["eos_token"],
                bos_token=fc["bos_token"],
            )
    except Exception as e:
        print(f"[WARN] formatter 초기화 실패: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 토큰 수 계산
# ══════════════════════════════════════════════════════════════════════════════

def count_tokens(llm: Llama, text: str) -> int:
    try:
        return len(llm.tokenize(text.encode("utf-8")))
    except Exception:
        return len(text) // 4


# ══════════════════════════════════════════════════════════════════════════════
# Round 1: Bio 추출 및 저장
# ══════════════════════════════════════════════════════════════════════════════

def round1_extract_and_save_bio(
    llm: Llama,
    formatter,
    config: Dict,
    embedding_fn: HuggingFaceEmbeddings,
    chroma_client: PersistentClient,
    dataset: List[Dict],
    output_dir: Path,
):
    print("\n" + "="*60)
    print("  ROUND 1: Bio 추출 및 ChromaDB 저장")
    print("="*60)

    bio_prompt = config.get("BIO_EXTRACTION_PROMPT_KOR", config.get("BIO_EXTRACTION_PROMPT", ""))
    max_tokens = 512

    round1_log = []

    for item in dataset:
        set_id = item["set_id"]
        utterances: List[str] = item["user_utterances"]
        print(f"\n[{set_id}] Bio 추출 중... ({len(utterances)}개 발화)")

        # 512 토큰 이내로 utterances를 조합
        combined = "\n".join(utterances)
        token_count = count_tokens(llm, combined)
        if token_count > max_tokens:
            print(f"  ⚠ 토큰 초과 ({token_count}), 앞부분 잘라냄")
            # 뒤에서부터 512 토큰 맞추기
            lines = utterances[:]
            while lines and count_tokens(llm, "\n".join(lines)) > max_tokens:
                lines.pop(0)
            combined = "\n".join(lines)

        messages_openai = convert_to_openai_messages([
            SystemMessage(bio_prompt),
            HumanMessage(combined),
        ])

        try:
            text_output = call_llm(llm, config, messages_openai, formatter)
            response = parse_llm_output(text_output)
            bio_list = parse_bio_with_importance(response.content) if response else []
        except Exception as e:
            print(f"  ✗ LLM 호출 실패: {e}")
            bio_list = []

        print(f"  → {len(bio_list)}개 bio 항목 추출됨")

        # ChromaDB 컬렉션에 저장
        collection_name = f"eval_{set_id.lower()}"
        try:
            # 컬렉션이 이미 있으면 삭제 후 재생성
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                pass
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "l2"},
            )

            for i, bio in enumerate(bio_list):
                content = bio.get("content", "")
                importance = bio.get("importance", 3)
                is_core = bio.get("is_core", False)
                if not content:
                    continue
                vector = embedding_fn.embed_query(content)
                collection.add(
                    ids=[f"{set_id}_bio_{i}"],
                    documents=[content],
                    embeddings=[vector],
                    metadatas=[{"importance": importance, "is_core": is_core, "set_id": set_id}],
                )
            print(f"  ✓ ChromaDB 컬렉션 '{collection_name}'에 {len(bio_list)}개 저장 완료")
        except Exception as e:
            print(f"  ✗ ChromaDB 저장 실패: {e}")

        round1_log.append({
            "set_id": set_id,
            "utterances_count": len(utterances),
            "bio_extracted": len(bio_list),
            "bio_items": [b.get("content", "") for b in bio_list],
        })

    # 추출 결과 로그 저장
    log_path = output_dir / "round1_extraction_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(round1_log, f, ensure_ascii=False, indent=2)
    print(f"\n[Round 1 완료] 로그 저장: {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Bio 검색 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

PRECISION_SYSTEM_PROMPT = """You are a personal assistant with memory of the user's profile and preferences.
Answer the following question with ONLY "Yes" or "No".
Do not provide any explanation or additional text — only output the single word "Yes" or "No".

Below is relevant information retrieved from memory:
{memory_context}
"""

IMPLICIT_SYSTEM_PROMPT = """You are a warm, helpful personal assistant who knows the user well.
Use the background knowledge about the user to give a personalized, natural response.
Do NOT explicitly say "I remember" or "Based on your memory" — just respond naturally as if you know the user.

User background knowledge:
{memory_context}
"""


def retrieve_bio_memories(
    chroma_client: PersistentClient,
    embedding_fn: HuggingFaceEmbeddings,
    set_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 1.15,
) -> Tuple[str, str]:
    """Set 컬렉션에서 쿼리와 유사한 bio 메모리 검색.
    Returns: (core_context, general_context)
    """
    collection_name = f"eval_{set_id.lower()}"
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        return "", ""

    # core bio
    try:
        core_data = collection.get(where={"is_core": True})
        core_docs = core_data.get("documents", [])
    except Exception:
        core_docs = []

    core_context = ""
    if core_docs:
        core_context = "### User Core Profile\n"
        for doc in core_docs:
            core_context += f"- {doc}\n"

    # general bio (유사도 검색)
    vector = embedding_fn.embed_query(query)
    try:
        result = collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return core_context, ""

    general_context = ""
    general_docs = []
    if result["documents"] and result["documents"][0]:
        for i in range(len(result["documents"][0])):
            dist = result["distances"][0][i]
            content = result["documents"][0][i]
            meta = result["metadatas"][0][i]
            if dist <= threshold and not meta.get("is_core", False):
                general_docs.append(content)

    if general_docs:
        general_context = "### Relevant User Preferences\n"
        for doc in general_docs:
            general_context += f"- {doc}\n"

    return core_context, general_context


# ══════════════════════════════════════════════════════════════════════════════
# Round 2: Precision Tasks
# ══════════════════════════════════════════════════════════════════════════════

def round2_precision_tasks(
    llm: Llama,
    formatter,
    config: Dict,
    chroma_client: PersistentClient,
    embedding_fn: HuggingFaceEmbeddings,
    dataset: List[Dict],
    output_dir: Path,
):
    print("\n" + "="*60)
    print("  ROUND 2: Precision Tasks (Yes/No)")
    print("="*60)

    csv_path = output_dir / "round2_precision_results.csv"
    stats: Dict[str, Dict] = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["set_id", "question", "expected_answer", "model_answer", "result"])

        for item in dataset:
            set_id = item["set_id"]
            tasks = item.get("precision_tasks", [])
            correct = 0
            total = len(tasks)
            print(f"\n[{set_id}] {total}개 질문 평가 중...")

            for task in tasks:
                question = task["question"]
                expected = task["answer"].strip()

                core_ctx, general_ctx = retrieve_bio_memories(
                    chroma_client, embedding_fn, set_id, question
                )
                memory_context = (core_ctx + "\n" + general_ctx).strip()

                system_msg = PRECISION_SYSTEM_PROMPT.format(memory_context=memory_context or "No memory available.")
                messages_openai = convert_to_openai_messages([
                    SystemMessage(system_msg),
                    HumanMessage(question),
                ])

                try:
                    raw_answer = call_llm(llm, config, messages_openai, formatter)
                    # Yes/No 파싱
                    model_answer = "Yes" if "yes" in raw_answer.lower()[:10] else "No"
                except Exception as e:
                    print(f"  ✗ 오류: {e}")
                    model_answer = "ERROR"

                is_correct = "correct" if model_answer == expected else "wrong"
                if model_answer != "ERROR" and is_correct == "correct":
                    correct += 1

                writer.writerow([set_id, question, expected, model_answer, is_correct])
                print(f"  Q: {question[:40]}... | 정답:{expected} | 모델:{model_answer} | {is_correct}")

            stats[set_id] = {"correct": correct, "total": total}

    # 통계 출력 및 저장
    total_correct = sum(v["correct"] for v in stats.values())
    total_questions = sum(v["total"] for v in stats.values())

    stats_path = output_dir / "round2_precision_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("  ROUND 2: Precision Tasks 결과 통계\n")
        f.write(f"  실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write("세트별 결과:\n")
        for set_id, s in stats.items():
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            f.write(f"  {set_id}: {s['correct']}/{s['total']} ({acc:.1f}%)\n")
        f.write("\n")
        overall_acc = total_correct / total_questions * 100 if total_questions > 0 else 0
        f.write(f"전체 결과: {total_correct}/{total_questions} ({overall_acc:.1f}%)\n")

    print(f"\n[Round 2 완료]")
    print(f"  CSV: {csv_path}")
    print(f"  통계: {stats_path}")
    print(f"  전체 정확도: {total_correct}/{total_questions} ({overall_acc:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Round 3: Implicit Generation Tasks
# ══════════════════════════════════════════════════════════════════════════════

def round3_implicit_tasks(
    llm: Llama,
    formatter,
    config: Dict,
    chroma_client: PersistentClient,
    embedding_fn: HuggingFaceEmbeddings,
    dataset: List[Dict],
    output_dir: Path,
):
    print("\n" + "="*60)
    print("  ROUND 3: Implicit Generation Tasks")
    print("="*60)

    csv_path = output_dir / "round3_implicit_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["set_id", "input", "expected_info", "model_response", "score(1-3)"])

        for item in dataset:
            set_id = item["set_id"]
            tasks = item.get("implicit_generation_tasks", [])
            print(f"\n[{set_id}] {len(tasks)}개 implicit 질문 생성 중...")

            for task in tasks:
                user_input = task["input"]
                expected_info = task["expected_info"]

                core_ctx, general_ctx = retrieve_bio_memories(
                    chroma_client, embedding_fn, set_id, user_input
                )
                memory_context = (core_ctx + "\n" + general_ctx).strip()

                system_msg = IMPLICIT_SYSTEM_PROMPT.format(memory_context=memory_context or "No memory available.")
                messages_openai = convert_to_openai_messages([
                    SystemMessage(system_msg),
                    HumanMessage(user_input),
                ])

                try:
                    model_response = call_llm(llm, config, messages_openai, formatter)
                except Exception as e:
                    print(f"  ✗ 오류: {e}")
                    model_response = "ERROR"

                writer.writerow([set_id, user_input, expected_info, model_response, ""])
                print(f"  Q: {user_input[:40]}... | 예상: {expected_info}")

    print(f"\n[Round 3 완료]")
    print(f"  CSV: {csv_path}")
    print(f"  ※ score(1-3) 컬럼은 수동 평가 후 입력해주세요.")


# ══════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bio Memory Evaluation Runner")
    parser.add_argument("--dataset", type=str, default="./dataset/PersoMem-Bench-Synthetic.json", help="평가 데이터셋 JSON 파일 경로")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="결과 저장 디렉토리 (기본: eval_results)")
    parser.add_argument("--rounds", type=str, default="1,2,3", help="실행할 라운드 (예: 1,2,3 또는 2,3)")
    parser.add_argument("--top_k", type=int, default=5, help="Bio 검색 top_k")
    parser.add_argument("--threshold", type=float, default=1.15, help="Bio 검색 유사도 임계치")
    args = parser.parse_args()

    rounds_to_run = [int(r.strip()) for r in args.rounds.split(",")]

    # 출력 디렉토리 설정 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = output_dir / "chromadb"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n결과 저장 위치: {output_dir}")

    # 데이터셋 로드
    with open(args.dataset, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset: List[Dict] = raw.get("data", raw) if isinstance(raw, dict) else raw
    print(f"데이터셋 로드 완료: {len(dataset)}개 세트")

    # 모델/설정 초기화
    print("\n모델 초기화 중...")
    config = load_config()
    llm = init_llm(config)
    formatter = init_formatter(config)
    embedding_fn = init_embedding(config)

    # ChromaDB 클라이언트 (평가 전용 경로)
    chroma_client = PersistentClient(path=str(chroma_dir))
    print(f"ChromaDB 경로: {chroma_dir}")

    # 설정 스냅샷 저장
    config_snapshot = {
        "dataset": args.dataset,
        "output_dir": str(output_dir),
        "rounds": rounds_to_run,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "timestamp": timestamp,
        "model_path": config["CHAT_MODEL_CONFIG"].get("model_path", ""),
    }
    with open(output_dir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    # 라운드 실행
    if 1 in rounds_to_run:
        round1_extract_and_save_bio(llm, formatter, config, embedding_fn, chroma_client, dataset, output_dir)

    if 2 in rounds_to_run:
        round2_precision_tasks(llm, formatter, config, chroma_client, embedding_fn, dataset, output_dir)

    if 3 in rounds_to_run:
        round3_implicit_tasks(llm, formatter, config, chroma_client, embedding_fn, dataset, output_dir)

    print(f"\n✅ 평가 완료! 결과 위치: {output_dir}")


if __name__ == "__main__":
    main()
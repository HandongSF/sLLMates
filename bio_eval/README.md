# Bio Memory Evaluation Suite

개인화 메모리(Bio) 기능 평가 스크립트 모음입니다.

## 파일 구성

```
bio_eval/
├── bio_eval_runner.py     # 메인 평가 실행기 (Round 1~3)
├── score_implicit.py      # Implicit Tasks 수동 평가 집계기
└── README.md
```

## 평가 구조

### Round 1 — Bio 추출 및 저장
- `BIO_EXTRACTION_PROMPT_KOR` + `user_utterances` → LLM 호출 (512 토큰 제한)
- 추출된 bio를 파싱하여 SET별 ChromaDB 컬렉션에 저장
- 출력: `round1_extraction_log.json`

### Round 2 — Precision Tasks (Yes/No 자동 채점)
- SET별 ChromaDB에서 유사도 검색으로 메모리 컨텍스트 구성
- `PRECISION_SYSTEM_PROMPT` + 메모리 + 질문 → LLM 호출
- 정답(Yes/No)과 비교하여 correct/wrong 기록
- 출력: `round2_precision_results.csv`, `round2_precision_stats.txt`

### Round 3 — Implicit Generation Tasks (수동 채점)
- SET별 ChromaDB에서 유사도 검색으로 메모리 컨텍스트 구성
- `IMPLICIT_SYSTEM_PROMPT` + 메모리 + 질문 → LLM 호출
- `score(1-3)` 컬럼은 비워두고 수동 평가 후 입력
- 출력: `round3_implicit_results.csv`

## 사용 방법

### 1. 전체 평가 실행

```bash
python bio_eval/bio_eval_runner.py \
  --dataset path/to/dataset.json \
  --output_dir eval_results
```

### 2. 특정 라운드만 실행

```bash
# Round 2, 3만 실행 (Round 1은 이미 완료된 경우)
python bio_eval/bio_eval_runner.py \
  --dataset path/to/dataset.json \
  --output_dir eval_results/run_20250101_120000 \
  --rounds 2,3
```

### 3. 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | 필수 | 데이터셋 JSON 파일 경로 |
| `--output_dir` | `eval_results` | 결과 저장 디렉토리 |
| `--rounds` | `1,2,3` | 실행할 라운드 (쉼표 구분) |
| `--top_k` | `5` | Bio 검색 top_k |
| `--threshold` | `1.15` | 유사도 임계치 (L2 거리) |

### 4. Implicit Tasks 수동 평가 후 집계

`round3_implicit_results.csv`의 `score(1-3)` 컬럼에 점수 입력 후:

```bash
python bio_eval/score_implicit.py \
  --csv eval_results/run_20250101_120000/round3_implicit_results.csv
```

## 점수 기준 (Implicit Tasks)

| 점수 | 기준 |
|------|------|
| 1 | 개인화 정보가 전혀 반영되지 않음 |
| 2 | 부분적으로 반영되었지만 자연스럽지 않음 |
| 3 | 개인화 정보를 잘 반영하여 자연스러운 응답 |

## 결과 디렉토리 구조

```
eval_results/
└── run_YYYYMMDD_HHMMSS/
    ├── chromadb/              # SET별 ChromaDB 컬렉션
    ├── eval_config.json       # 실행 설정 스냅샷
    ├── round1_extraction_log.json
    ├── round2_precision_results.csv
    ├── round2_precision_stats.txt
    ├── round3_implicit_results.csv
    └── round3_implicit_score_summary.txt  # score_implicit.py 실행 후 생성
```
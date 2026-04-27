"""
Implicit Generation Tasks 수동 평가 집계 스크립트

사용법:
  python bio_eval/score_implicit.py --csv path/to/round3_implicit_results.csv

round3_implicit_results.csv 의 score(1-3) 컬럼에 점수를 입력한 후 실행하면
세트별 / 전체 통계를 출력하고 .txt 파일로 저장합니다.

점수 기준 (1~3):
  1 = 개인화 반영 없음 / 틀림
  2 = 부분적으로 반영
  3 = 개인화 정보를 잘 반영하여 자연스러운 응답
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def aggregate_scores(csv_path: Path) -> None:
    set_scores: dict = defaultdict(list)
    skipped_rows = 0
    total_rows = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            set_id = row.get("set_id", "UNKNOWN").strip()
            score_str = row.get("score(1-3)", "").strip()

            if not score_str:
                skipped_rows += 1
                continue

            try:
                score = int(score_str)
                if score not in (1, 2, 3):
                    raise ValueError(f"유효하지 않은 점수: {score_str}")
                set_scores[set_id].append(score)
            except ValueError as e:
                print(f"  [WARN] 행 무시 ({set_id}): {e}")
                skipped_rows += 1

    if skipped_rows > 0:
        print(f"\n⚠  {skipped_rows}개 행이 비어 있거나 유효하지 않아 제외되었습니다.")

    if not set_scores:
        print("평가할 데이터가 없습니다. score(1-3) 컬럼을 확인하세요.")
        return

    max_score_per_question = 3
    lines = []

    lines.append("=" * 60)
    lines.append("  Implicit Generation Tasks 수동 평가 결과")
    lines.append(f"  집계 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  파일: {csv_path.name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("세트별 결과:")

    grand_total_score = 0
    grand_total_max = 0

    for set_id in sorted(set_scores.keys()):
        scores = set_scores[set_id]
        total_score = sum(scores)
        max_score = len(scores) * max_score_per_question
        pct = total_score / max_score * 100 if max_score > 0 else 0
        line = f"  {set_id}: {total_score}/{max_score} ({pct:.1f}%)  [평균: {total_score/len(scores):.2f}]"
        lines.append(line)
        grand_total_score += total_score
        grand_total_max += max_score

    lines.append("")
    overall_pct = grand_total_score / grand_total_max * 100 if grand_total_max > 0 else 0
    lines.append(f"전체 합계: {grand_total_score}/{grand_total_max} ({overall_pct:.1f}%)")
    lines.append(f"전체 평균 점수: {grand_total_score / (grand_total_max / max_score_per_question):.2f} / {max_score_per_question}")

    output_str = "\n".join(lines)
    print("\n" + output_str)

    # 같은 디렉토리에 .txt 저장
    out_path = csv_path.parent / "round3_implicit_score_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_str + "\n")
    print(f"\n✅ 통계 저장 완료: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Implicit Tasks 수동 평가 집계")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="round3_implicit_results.csv 경로",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"파일을 찾을 수 없습니다: {csv_path}")
        return

    aggregate_scores(csv_path)


if __name__ == "__main__":
    main()
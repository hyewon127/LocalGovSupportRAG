# [WBS 8.2] 검색 품질 게이트 - 코드를 바꾼 뒤에도 검색 성능이 분석모델 정의서 목표(Hit@5 80%, MRR 0.6) 이상인지
#
# 실행 (OpenSearch 실행 중, 프로젝트 루트에서, 약 10초):
#   .venv\Scripts\python.exe tests\test_retrieval_quality.py
#
# [왜 필요한가] 다른 테스트는 "에러 없이 동작하는가"를 보지만, 검색 가중치·필터·청크 상한 같은 걸 바꾸면
#   에러 없이 "조용히 나빠질" 수 있습니다. WBS 7.2 평가 스크립트의 운영 설정을 그대로 돌려서 목표치 아래로
#   떨어지면 실패시킵니다 (성능 회귀 테스트). 원 질의뿐 아니라, 서비스 성능에 더 가까운 어휘 치환 질의도 목표를 지켜야 합니다.
#   - 2026-09-22 기준값: 원 질의 Hit@5 100% / MRR 0.927, 어휘 치환 88.9% / 0.713 (docs/evaluation_report.md)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

import json  # noqa: E402

from retrieval_eval import CONFIGS, EVAL_SET_PATH, PRODUCTION, evaluate_config  # noqa: E402  (src/rag, src/indexing 경로도 잡아줌)
from config import get_client  # noqa: E402
from llm_client import get_llm_client  # noqa: E402
from query_slots import extract_slots, fetch_candidate_values  # noqa: E402

TARGET_HIT5 = 0.80  # 분석모델 정의서 2.5
TARGET_MRR = 0.60
results: list[tuple[str, bool]] = []


def check(name: str, condition, detail="") -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""))


def main() -> int:
    client = get_client()
    candidates = fetch_candidate_values(client)
    llm = get_llm_client()
    queries = [q for q in json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))["queries"] if q["type"] == "on_topic"]
    production = next(c for c in CONFIGS if c["name"] == PRODUCTION)

    for key, label in (("question", "원 질의"), ("paraphrase", "어휘 치환 질의")):
        slots = {q["id"]: extract_slots(q[key], candidates=candidates, llm_client=llm) for q in queries}
        s = evaluate_config(production, queries, slots, client, question_key=key)["summary"]
        check(f"{label}: Hit@5 >= {TARGET_HIT5:.0%}", s["hit@5"] >= TARGET_HIT5, f"{s['hit@5']:.1%}")
        check(f"{label}: MRR@5 >= {TARGET_MRR}", s["mrr@5"] >= TARGET_MRR, f"{s['mrr@5']:.3f}")

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

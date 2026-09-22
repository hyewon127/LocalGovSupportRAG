# [WBS 7.2~7.3] 평가 지표 계산 단위 테스트 - OpenSearch/LLM 없이 몇 초 안에 끝나는 순수 계산 검증
#
# 실행: .venv\Scripts\python.exe tests\test_evaluation_metrics.py
#
# 왜 필요한가: 리포트에 실릴 Hit@5/MRR/인용률 숫자는 전부 src/evaluation/metrics.py를 거칩니다.
#   예를 들어 순위를 0부터 세는 실수 하나로 MRR이 전부 두 배가 되어도, 평가 스크립트 결과만 봐서는 알아채기 어렵습니다.
#   그래서 손으로 계산할 수 있는 작은 예시로 기대값을 먼저 고정해 둡니다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))

from metrics import (  # noqa: E402
    citation_stats,
    first_relevant_rank,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    unsupported_numbers,
)

results: list[tuple[str, bool]] = []


def check(name: str, actual, expected) -> None:
    ok = actual == expected if not isinstance(expected, float) else abs(actual - expected) < 1e-9
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  (실제 {actual!r} / 기대 {expected!r})")


def main() -> int:
    # 청크 순서 [A, A, B, C, D] - 같은 사업 A가 1·2번째를 차지한 상황
    retrieved = ["A", "A", "B", "C", "D"]
    check("순위: 1부터 센다(A는 1등)", first_relevant_rank(retrieved, {"A"}, 5), 1)
    check("순위: 청크 기준(B는 사업으로 2등이지만 청크로 3번째)", first_relevant_rank(retrieved, {"B"}, 5), 3)
    check("순위: 여러 정답이면 가장 앞의 것", first_relevant_rank(retrieved, {"D", "C"}, 5), 4)
    check("순위: k 밖은 없음 처리", first_relevant_rank(retrieved, {"D"}, 3), None)

    check("Hit@1: 1등이 정답이면 1", hit_at_k(retrieved, {"A"}, 1), 1.0)
    check("Hit@1: 3등 정답은 0", hit_at_k(retrieved, {"B"}, 1), 0.0)
    check("Hit@3: 3등 정답은 1", hit_at_k(retrieved, {"B"}, 3), 1.0)
    check("Hit@5: 정답 없으면 0", hit_at_k(retrieved, {"Z"}, 5), 0.0)

    check("RR: 1등 = 1.0", reciprocal_rank(retrieved, {"A"}, 5), 1.0)
    check("RR: 3등 = 1/3", reciprocal_rank(retrieved, {"B"}, 5), 1 / 3)
    check("RR: k 밖 = 0", reciprocal_rank(retrieved, {"D"}, 3), 0.0)
    check("MRR: (1 + 1/3 + 0) / 3", mean([reciprocal_rank(retrieved, r, 5) for r in ({"A"}, {"B"}, {"Z"})]),
          (1 + 1 / 3) / 3)

    check("Recall@5: 정답 4개 중 2개 포함", recall_at_k(retrieved, {"B", "C", "Y", "Z"}, 5), 0.5)
    check("Recall@5: 중복 청크는 한 번만 셈", recall_at_k(retrieved, {"A"}, 5), 1.0)
    check("Recall: 정답이 없는 질의는 0", recall_at_k(retrieved, set(), 5), 0.0)
    check("mean: 빈 목록은 0", mean([]), 0.0)

    s = citation_stats("A 사업을 추천합니다 [1][3]. 없는 번호 [7]과 [0].", num_sources=5)
    check("인용: 유효 번호", s["cited"], [1, 3])
    check("인용: 없는 번호([0], [7])", s["invalid"], [0, 7])
    check("인용: 유효 인용 있음", s["has_valid_citation"], True)
    check("인용: 인용 없는 답변", citation_stats("근거 없이 쓴 답변", 5)["has_valid_citation"], False)

    context = ["지원금액: 최대 5,000만원, 자부담 20% / 선정규모 30개사"]
    check("수치: 근거와 같은 값은 통과(쉼표·공백 차이 무시)", unsupported_numbers("최대 5000만 원, 자부담 20 %", context), [])
    check("수치: 근거에 없는 금액은 검출", unsupported_numbers("최대 1억원을 지원하며 30개사를 뽑습니다", context), ["1억원"])
    check("수치: 수치 없는 답변은 검출 없음", unsupported_numbers("자세한 내용은 공고를 확인하세요", context), [])
    # 공고문에 흔한 "숫자+한글 단위" 혼합 표기 (처음 패턴이 못 잡았던 형식)
    context_mixed = ["지원한도: 업체당 6백만원, 총 3천만원"]
    check("수치: '6백만원'·'3천만원' 표기도 근거와 대조", unsupported_numbers("6백만원씩 총 3천만원", context_mixed), [])
    check("수치: 근거에 없는 '9백만원'은 검출", unsupported_numbers("최대 9백만원 지원", context_mixed), ["9백만원"])

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

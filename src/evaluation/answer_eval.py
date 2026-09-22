# [WBS 7.3] 출처 인용률 / hallucination 비율 체크 + 근거 없음 가드레일 평가 + 응답 시간
#
# 실행 (OpenSearch 실행 중, 프로젝트 루트에서):
#   .venv\Scripts\python.exe src\evaluation\answer_eval.py
#   -> 콘솔 출력 + docs/eval/answer_results.json 저장
#
# [세 부분으로 나눈 이유 - 무엇이 LLM 없이 측정 가능한가]
#   A. 근거 없음 가드레일(guardrail.py [호출 전]) - LLM 없이 측정 가능. 관련 질의는 통과시키고, 무관 질의는
#      막는지 봅니다. WBS 5.6에서 질의 10개로 잡은 임계값 0.77을, 그 10개와 겹치지 않는 이 평가셋으로
#      다시 확인하는 것이 목적입니다(guardrail.py [한계] 주석에서 예고한 재보정).
#   B. 출처 인용률 / hallucination 대리 지표 - 실제 LLM 답변이 있어야 측정 가능. 키가 없으면 "측정 안 함"으로
#      기록하고, 수치를 지어내지 않습니다. 키를 넣고 다시 실행하면 그대로 측정됩니다.
#      (코드 경로 자체는 tests/test_answer_eval.py가 가짜 LLM으로 검증)
#   C. 응답 시간 - 검색 구간은 항상 측정, LLM 구간은 B를 돌릴 때만 측정.
#
# [지표 정의] (분석모델 정의서 2.5)
#   출처 인용률(목표 95% 이상) = 모델이 답변한 경우 중, 유효한 [번호] 인용이 1개 이상 있는 답변 비율.
#     모델이 스스로 "찾지 못했습니다"라고 답한 경우는 인용할 대상이 없으므로 분모에서 뺍니다.
#     가드레일 적용 "전" 원본 답변으로 잽니다 - 적용 후 답변은 인용 없는 답을 이미 버렸기 때문에 항상 100%가 됩니다.
#   Hallucination 비율(목표 5% 이하) = 대리 지표: 인용한 근거에 없는 수치(금액/비율/인원 등)가 1개 이상 있는 답변 비율.
#     한계는 metrics.unsupported_numbers() 주석 참고 (수치 없는 지어낸 문장은 못 잡음, 표기 차이는 오탐 가능).

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
for _module_dir in (_PROJECT_ROOT / "src" / "rag", _PROJECT_ROOT / "src" / "indexing"):
    if str(_module_dir) not in sys.path:
        sys.path.insert(0, str(_module_dir))

from config import check_connection, get_client  # noqa: E402
from context_builder import build_context  # noqa: E402
from guardrail import KNN_RELEVANCE_THRESHOLD  # noqa: E402
from hybrid_search import hybrid_search  # noqa: E402
from llm_client import LLM_MODEL, LLM_PROVIDER, get_llm_client  # noqa: E402
from metrics import citation_stats, mean, unsupported_numbers  # noqa: E402
from pipeline import answer_question  # noqa: E402
from prompt_template import NO_EVIDENCE_MESSAGE  # noqa: E402
from query_slots import extract_slots, fetch_candidate_values  # noqa: E402

EVAL_SET_PATH = _PROJECT_ROOT / "data" / "eval" / "eval_queries.json"
RESULT_PATH = _PROJECT_ROOT / "docs" / "eval" / "answer_results.json"
THRESHOLDS = [round(0.70 + 0.01 * i, 2) for i in range(16)]  # 0.70 ~ 0.85


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(len(values) * 0.95) - 1)] if values else 0.0


# ── A. 근거 없음 가드레일 ────────────────────────────────────────────
def collect_guardrail_scores(queries: list[dict], client, candidates: dict, llm) -> list[dict]:
    """
    질의마다 운영 경로(슬롯 추출 -> 하이브리드 검색)를 그대로 돌려서, 가드레일이 보는 값(top-5 중 최고 kNN 점수)을 모읍니다.
    관련 질의는 원 질의와 어휘 치환 질의를 둘 다 넣습니다 - 실제 사용자 표현(어휘 치환)에서 점수가 낮아져
    멀쩡한 질문이 "근거 없음"으로 막히는지도 봐야 하기 때문입니다.

    pipeline.answer_question()을 쓰지 않고 앞 단계만 직접 부르는 이유: answer_question은 LLM 키가 있으면
    답변 생성까지 가서 비용과 시간이 들고, 여기서 필요한 건 가드레일 직전의 점수뿐입니다.
    """
    records = []
    for q in queries:
        variants = [("off_topic", q["question"])] if q["type"] == "off_topic" else \
            [("original", q["question"]), ("paraphrase", q["paraphrase"])]
        for variant, text in variants:
            started = time.perf_counter()
            slots = extract_slots(text, candidates=candidates, llm_client=llm)
            chunks = hybrid_search(text, region=slots["region"], categories=slots["categories"],
                                   target_keywords=slots["target_keywords"], client=client)
            elapsed_ms = (time.perf_counter() - started) * 1000
            knn_scores = [c["knn_score"] for c in chunks if c.get("knn_score") is not None]
            records.append({"id": q["id"], "variant": variant, "text": text,
                            "max_knn_score": max(knn_scores) if knn_scores else None,
                            "retrieval_ms": elapsed_ms})
    return records


def guardrail_table(records: list[dict]) -> list[dict]:
    """임계값별로 관련 질의 통과율(높을수록 좋음)과 무관 질의 차단율(높을수록 좋음)을 계산합니다."""
    def passes(r, t):
        return r["max_knn_score"] is not None and r["max_knn_score"] >= t

    groups = {v: [r for r in records if r["variant"] == v] for v in ("original", "paraphrase", "off_topic")}
    rows = []
    for t in THRESHOLDS:
        rows.append({
            "threshold": t,
            "on_topic_pass_original": mean([1.0 if passes(r, t) else 0.0 for r in groups["original"]]),
            "on_topic_pass_paraphrase": mean([1.0 if passes(r, t) else 0.0 for r in groups["paraphrase"]]),
            "off_topic_block": mean([0.0 if passes(r, t) else 1.0 for r in groups["off_topic"]]),
        })
    return rows


# ── B. 출처 인용률 / hallucination 대리 지표 ─────────────────────────
def evaluate_answers(queries: list[dict], client, candidates: dict, llm) -> dict:
    """
    실제 LLM으로 답변을 생성해서 원본 답변(가드레일 적용 전)의 인용과 수치를 검사합니다.
    llm은 None이 아니어야 합니다 (호출하는 쪽에서 키 유무를 먼저 확인).
    """
    per_query = []
    for q in queries:
        started = time.perf_counter()
        result = answer_question(q["question"], os_client=client, llm_client=llm, candidates=candidates, return_trace=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        raw = result["trace"]["raw_answer"]
        chunks = result["trace"]["chunks"] or []
        record = {"id": q["id"], "type": q["type"], "question": q["question"], "final_status": result["status"],
                  "raw_answer": raw, "latency_ms": elapsed_ms, "generated": raw is not None}
        if raw is not None:
            stats = citation_stats(raw, num_sources=len(chunks))
            # 인용한 근거의 원문 = LLM이 실제로 받은 컨텍스트 블록. build_context()를 청크 1개씩 다시 불러
            # 프롬프트에 들어간 것과 똑같은 텍스트(사업명/지원대상/접수기간/지원금액/본문)를 만듭니다.
            cited_texts = [build_context([chunks[n - 1]])["context_text"] for n in stats["cited"]]
            record.update({
                "self_declared_no_evidence": NO_EVIDENCE_MESSAGE in raw,
                "cited": stats["cited"],
                "invalid_citations": stats["invalid"],
                "has_valid_citation": stats["has_valid_citation"],
                "unsupported_numbers": unsupported_numbers(raw, cited_texts) if stats["cited"] else [],
            })
        per_query.append(record)

    answered = [r for r in per_query if r["generated"] and not r["self_declared_no_evidence"]]
    off_topic = [r for r in per_query if r["type"] == "off_topic"]
    generation_latencies = [r["latency_ms"] for r in per_query if r["generated"]]
    return {
        "measured": True,
        "llm": f"{LLM_PROVIDER}/{LLM_MODEL}",
        "summary": {
            "num_generated": sum(r["generated"] for r in per_query),
            "num_answered": len(answered),
            "citation_rate": mean([1.0 if r["has_valid_citation"] else 0.0 for r in answered]),
            "invalid_citation_answer_rate": mean([1.0 if r["invalid_citations"] else 0.0 for r in answered]),
            "hallucination_proxy_rate": mean([1.0 if r["unsupported_numbers"] else 0.0 for r in answered]),
            "off_topic_correct_refusal": mean([1.0 if r["final_status"] == "no_evidence" else 0.0 for r in off_topic]),
            "final_status_counts": {s: sum(r["final_status"] == s for r in per_query)
                                    for s in sorted({r["final_status"] for r in per_query})},
            "latency_ms_mean_with_llm": mean(generation_latencies),
            "latency_ms_p95_with_llm": _p95(generation_latencies),
        },
        "per_query": per_query,
    }


# ── 출력 ─────────────────────────────────────────────────────────────
def print_report(records: list[dict], table: list[dict], answers: dict) -> None:
    print("\n### A. 근거 없음 가드레일 - 질의별 최고 kNN 점수")
    for variant, label in (("original", "관련(원 질의)"), ("paraphrase", "관련(어휘 치환)"), ("off_topic", "무관")):
        scores = [r["max_knn_score"] for r in records if r["variant"] == variant and r["max_knn_score"] is not None]
        print(f"- {label}: 최소 {min(scores):.3f} / 중앙값 {statistics.median(scores):.3f} / 최대 {max(scores):.3f} (n={len(scores)})")

    print("\n| 임계값 | 관련 질의 통과(원) | 관련 질의 통과(어휘 치환) | 무관 질의 차단 |\n|---|---|---|---|")
    for row in table:
        mark = " ← 현재" if abs(row["threshold"] - KNN_RELEVANCE_THRESHOLD) < 1e-9 else ""
        print(f"| {row['threshold']:.2f}{mark} | {row['on_topic_pass_original']:.1%} | "
              f"{row['on_topic_pass_paraphrase']:.1%} | {row['off_topic_block']:.1%} |")

    wrong = [r for r in records if (r["variant"] == "off_topic") == (r["max_knn_score"] is not None and r["max_knn_score"] >= KNN_RELEVANCE_THRESHOLD)]
    print(f"\n현재 임계값({KNN_RELEVANCE_THRESHOLD})에서 잘못 판정된 질의 {len(wrong)}건:")
    for r in wrong:
        verdict = "무관인데 통과" if r["variant"] == "off_topic" else "관련인데 차단"
        print(f"- [{verdict}] {r['id']}({r['variant']}) {r['max_knn_score']:.3f}  {r['text']}")

    print("\n### B. 출처 인용률 / hallucination 대리 지표")
    if not answers["measured"]:
        print(f"- 측정 안 함: {answers['reason']}")
    else:
        s = answers["summary"]
        print(f"- LLM: {answers['llm']} / 답변 생성 {s['num_generated']}건 (자체 '근거 없음' 제외 {s['num_answered']}건)")
        print(f"- 출처 인용률: {s['citation_rate']:.1%} (목표 95% 이상)")
        print(f"- 없는 번호 인용한 답변: {s['invalid_citation_answer_rate']:.1%}")
        print(f"- 근거에 없는 수치 포함 답변(hallucination 대리): {s['hallucination_proxy_rate']:.1%} (목표 5% 이하)")
        print(f"- 무관 질의 올바른 거절: {s['off_topic_correct_refusal']:.1%} / 최종 상태 분포 {s['final_status_counts']}")
        print(f"- 응답 시간(LLM 포함): 평균 {s['latency_ms_mean_with_llm'] / 1000:.2f}초 / p95 {s['latency_ms_p95_with_llm'] / 1000:.2f}초")

    retrieval = [r["retrieval_ms"] for r in records]
    print(f"\n### C. 응답 시간 (검색 구간: 슬롯 추출 + 하이브리드 검색)\n- 중앙값 {statistics.median(retrieval):.0f}ms / p95 {_p95(retrieval):.0f}ms (n={len(retrieval)})")


def main() -> int:
    client = get_client()
    if not check_connection(client):
        return 1
    dataset = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    queries = dataset["queries"]
    candidates = fetch_candidate_values(client)
    llm = get_llm_client()

    records = collect_guardrail_scores(queries, client, candidates, llm)
    table = guardrail_table(records)

    if llm is None:
        answers = {"measured": False,
                   "reason": f"LLM API 키가 없어 답변을 생성할 수 없음 (LLM_PROVIDER={LLM_PROVIDER}). "
                             ".env에 키를 넣고 이 스크립트를 다시 실행하면 측정됩니다."}
    else:
        answers = evaluate_answers(queries, client, candidates, llm)

    print_report(records, table, answers)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_set_version": dataset["version"],
        "current_threshold": KNN_RELEVANCE_THRESHOLD,
        "guardrail": {"records": records, "threshold_table": table},
        "answers": answers,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {RESULT_PATH.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

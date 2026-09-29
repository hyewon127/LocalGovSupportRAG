# [WBS 7.2] Hit@k / MRR 측정 - 테스트 질의셋(data/eval/eval_queries.json)으로 검색 성능을 수치화
#
# 실행 (OpenSearch 실행 중, 프로젝트 루트에서):
#   .venv\Scripts\python.exe src\evaluation\retrieval_eval.py
#   -> 콘솔에 마크다운 표 출력 + docs/eval/retrieval_results.json 저장 (7.4 리포트의 근거 데이터)
#
# [무엇을 비교하는가] 분석모델 정의서가 요구하는 비교를 그대로 설정(CONFIGS)으로 만들었습니다.
#   - 슬라이드 10 "하이브리드 vs 단일 검색 방식 비교"  -> hybrid / bm25_only / knn_only
#   - 슬라이드 4 하이브리드 가중치 "탐색 범위 0.2~0.8" -> w0.2 ~ w0.8
#   - 추가로, 지금까지 "좋을 것 같아서" 넣었던 설계 결정 2가지가 실제로 효과가 있었는지 확인합니다.
#       no_filter    : 슬롯(지역/분야) 필터를 끄면? (query_slots.py가 도움이 되는가)
#       no_diversity : 사업당 2청크 상한을 끄면? (hybrid_search.py MAX_CHUNKS_PER_PROGRAM이 도움이 되는가)
#   설계 결정을 "했다"에서 끝내지 않고 "측정해서 효과를 확인했다"까지 가는 게 이 단계의 핵심입니다.
#
# [지표] 정의서 2.5의 목표 지표 Hit@5(80% 이상), MRR(0.6 이상) + 보조 지표 Hit@1, Hit@3, Recall@5.
#   순위 기준은 metrics.py 상단 [순위의 기준] 참고 (LLM에 넘어가는 청크 순서에서 사업이 처음 나온 위치).
#
# [한계 - 리포트에 함께 적어야 할 것]
#   - 질의 27개는 통계적으로 작은 표본입니다. 질의 1개가 Hit@5를 3.7%p 움직이므로, 설정 간 1~2개 차이는 우연일 수 있습니다.
#   - 정답 라벨은 사람(초안: Claude)이 판정한 것이라 판정 기준에 따라 점수가 달라집니다 (eval_queries.json의 labeling_protocol).

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
from hybrid_search import hybrid_search  # noqa: E402
from llm_client import get_llm_client  # noqa: E402
from metrics import first_relevant_rank, hit_at_k, mean, recall_at_k, reciprocal_rank  # noqa: E402
from query_slots import extract_slots, fetch_candidate_values  # noqa: E402

EVAL_SET_PATH = _PROJECT_ROOT / "data" / "eval" / "eval_queries.json"
RESULT_PATH = _PROJECT_ROOT / "docs" / "eval" / "retrieval_results.json"
TOP_K = 5  # 분석모델 정의서 1.3 "검색 top-k 권장값 5" = LLM에 넘기는 근거 수

# name: 결과 표에 쓸 이름 / use_filters: 슬롯 필터 적용 여부 / 나머지는 hybrid_search() 인자
PRODUCTION = "hybrid"
CONFIGS = [
    {"name": "hybrid", "label": "하이브리드 0.4:0.6 (운영 설정)", "use_filters": True,
     "bm25_weight": 0.4, "knn_weight": 0.6, "max_chunks_per_program": 2},
    {"name": "bm25_only", "label": "BM25 단독", "use_filters": True,
     "bm25_weight": 1.0, "knn_weight": 0.0, "max_chunks_per_program": 2},
    {"name": "knn_only", "label": "kNN 단독", "use_filters": True,
     "bm25_weight": 0.0, "knn_weight": 1.0, "max_chunks_per_program": 2},
    {"name": "no_filter", "label": "하이브리드, 슬롯 필터 끔", "use_filters": False,
     "bm25_weight": 0.4, "knn_weight": 0.6, "max_chunks_per_program": 2},
    {"name": "no_diversity", "label": "하이브리드, 사업당 상한 끔", "use_filters": True,
     "bm25_weight": 0.4, "knn_weight": 0.6, "max_chunks_per_program": None},
] + [
    {"name": f"w{w:.1f}", "label": f"BM25:kNN = {w:.1f}:{1 - w:.1f}", "use_filters": True,
     "bm25_weight": w, "knn_weight": round(1 - w, 1), "max_chunks_per_program": 2, "sweep": True}
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
]


# [어휘 치환 스트레스 테스트] 원 질의는 사업명 목록을 본 뒤 작성해서 제목 단어와 겹치는 편향이 있습니다
# (27개 중 20개가 정답 사업명의 단어를 그대로 포함). 그래서 같은 의도를 제목 단어를 피해 다시 쓴 문장
# (eval_queries.json의 paraphrase)으로 검색 방식 3가지를 한 번 더 비교합니다. 실제 사용자는 공고 제목을 모르고
# 자기 말로 묻기 때문에, 이 결과가 서비스 성능에 더 가까운 추정치입니다.
PARAPHRASE_CONFIGS = ["hybrid", "bm25_only", "knn_only"]


def evaluate_config(config: dict, queries: list[dict], slots_by_id: dict, client, question_key: str = "question") -> dict:
    per_query = []
    latencies_ms = []
    for q in queries:
        slots = slots_by_id[q["id"]]
        relevant = set(q["relevant_program_ids"])
        started = time.perf_counter()
        chunks = hybrid_search(
            q[question_key],
            region=slots["region"] if config["use_filters"] else None,
            categories=slots["categories"] if config["use_filters"] else None,
            # target_keywords는 필터가 아니라 BM25 검색어 힌트라서(query_slots.py [설계 결정 1]) 필터 on/off와
            # 무관하게 넣습니다 - no_filter 설정은 "필터"의 효과만 따로 보려는 것이므로.
            target_keywords=slots["target_keywords"],
            top_k=TOP_K,
            client=client,
            bm25_weight=config["bm25_weight"],
            knn_weight=config["knn_weight"],
            max_chunks_per_program=config["max_chunks_per_program"],
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        retrieved = [c["program_id"] for c in chunks]
        per_query.append({
            "id": q["id"],
            "question": q[question_key],
            "slots": slots,
            "retrieved_program_ids": retrieved,
            "retrieved_program_names": [c["program_name"] for c in chunks],
            "max_knn_score": max((c["knn_score"] for c in chunks if c.get("knn_score") is not None), default=None),
            "first_relevant_rank": first_relevant_rank(retrieved, relevant, TOP_K),
            "hit@1": hit_at_k(retrieved, relevant, 1),
            "hit@3": hit_at_k(retrieved, relevant, 3),
            "hit@5": hit_at_k(retrieved, relevant, 5),
            "rr@5": reciprocal_rank(retrieved, relevant, 5),
            "recall@5": recall_at_k(retrieved, relevant, 5),
            "distinct_programs": len(set(retrieved)),
        })

    summary = {
        "hit@1": mean([r["hit@1"] for r in per_query]),
        "hit@3": mean([r["hit@3"] for r in per_query]),
        "hit@5": mean([r["hit@5"] for r in per_query]),
        "mrr@5": mean([r["rr@5"] for r in per_query]),
        "recall@5": mean([r["recall@5"] for r in per_query]),
        "avg_distinct_programs": mean([r["distinct_programs"] for r in per_query]),
        # 첫 질의는 임베딩 모델 로딩 시간이 섞일 수 있어 중앙값/95퍼센타일을 같이 봅니다.
        "latency_ms_median": statistics.median(latencies_ms),
        "latency_ms_p95": sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1],
    }
    return {"config": config, "summary": summary, "per_query": per_query}


def _row(r: dict) -> str:
    s = r["summary"]
    return (f"| {r['config']['label']} | {s['hit@1']:.1%} | {s['hit@3']:.1%} | {s['hit@5']:.1%} | "
            f"{s['mrr@5']:.3f} | {s['recall@5']:.1%} | {s['avg_distinct_programs']:.1f} |")


_HEADER = "| 설정 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | top-5 내 사업 수 |\n|---|---|---|---|---|---|---|"


def _print_misses(title: str, result: dict) -> None:
    misses = [q for q in result["per_query"] if q["hit@5"] == 0]
    print(f"\n### {title} ({len(misses)}건)")
    for q in misses:
        print(f"- {q['id']} {q['question']}  (슬롯 {q['slots']})")
        for name in q["retrieved_program_names"]:
            print(f"    · {name[:60]}")


def print_tables(results: list[dict], paraphrase_results: list[dict]) -> None:
    print("\n### 검색 방식 / 설계 결정 비교 (원 질의)\n" + _HEADER)
    for r in results:
        if not r["config"].get("sweep"):
            print(_row(r))
    print("\n### 하이브리드 가중치 탐색 (정의서 탐색 범위 0.2~0.8, 원 질의)\n" + _HEADER)
    for r in results:
        if r["config"].get("sweep"):
            print(_row(r))
    print("\n### 어휘 치환 질의 (공고 제목 단어를 피해 다시 쓴 문장)\n" + _HEADER)
    for r in paraphrase_results:
        print(_row(r))

    _print_misses("운영 설정 - 원 질의에서 top-5 안에 정답을 못 찾은 질의",
                  next(r for r in results if r["config"]["name"] == PRODUCTION))
    _print_misses("운영 설정 - 어휘 치환 질의에서 top-5 안에 정답을 못 찾은 질의",
                  next(r for r in paraphrase_results if r["config"]["name"] == PRODUCTION))


def main() -> int:
    client = get_client()
    if not check_connection(client):
        return 1

    dataset = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    queries = [q for q in dataset["queries"] if q["type"] == "on_topic"]

    # 슬롯은 설정과 무관하게 질의마다 한 번만 추출해서 모든 설정에 똑같이 씁니다 - 설정 간 차이가
    # 슬롯 추출의 흔들림(LLM 폴백 등)이 아니라 검색 방식 차이에서만 나오게 하기 위함입니다.
    candidates = fetch_candidate_values(client)
    llm = get_llm_client()
    slots_by_id = {q["id"]: extract_slots(q["question"], candidates=candidates, llm_client=llm) for q in queries}

    results = [evaluate_config(config, queries, slots_by_id, client) for config in CONFIGS]

    # 바꿔 쓴 문장은 슬롯도 그 문장에서 다시 뽑습니다 - 실제 서비스에서도 사용자가 쓴 문장에서 추출하므로,
    # "제목 단어가 없어서 슬롯이 덜 잡히는 효과"까지 포함해서 측정해야 공정합니다.
    paraphrase_slots = {q["id"]: extract_slots(q["paraphrase"], candidates=candidates, llm_client=llm) for q in queries}
    paraphrase_results = [
        evaluate_config(config, queries, paraphrase_slots, client, question_key="paraphrase")
        for config in CONFIGS if config["name"] in PARAPHRASE_CONFIGS
    ]
    print_tables(results, paraphrase_results)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_set_version": dataset["version"],
        "num_queries": len(queries),
        "top_k": TOP_K,
        "slot_extraction": "정규식 + LLM 폴백" if llm else "정규식만 (LLM 키 없음 - query_slots.py LLM 폴백 미사용)",
        # 가중치 탐색(sweep) 7개는 요약만 저장합니다 - 질의별 상세까지 전부 저장하면 파일이 550KB를 넘는데,
        # 실패 분석에 필요한 건 운영 설정/단일 방식/설계 결정 비교의 질의별 결과뿐이라서입니다.
        "results": [{**r, "per_query": None} if r["config"].get("sweep") else r for r in results],
        "paraphrase_results": paraphrase_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {RESULT_PATH.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# [WBS 5.2] 하이브리드 검색 쿼리 작성 (BM25+kNN 가중합)
#
# 배경: src/indexing/filter_search_test.py(WBS 4.6)에서 BM25 검색과 kNN 검색이 "각각" 필터와
# 함께 동작하는 것까지만 확인했습니다. 이 파일이 하는 일은 그 둘을 "하나의 순위"로 합치는 것입니다.
#
# 분석모델 정의서.pptx(2.3 하이브리드 검색 설계)에 이미 방식이 정해져 있어서, 새로 설계하지 않고
# 문서에 적힌 값을 그대로 상수로 반영했습니다:
#   - 채택 방식: "두 점수를 가중합하여 재정렬" (BM25/kNN 단독이 아니라 최종 채택은 하이브리드)
#     이유: "지자체명/분야는 정확한 키워드 매칭이 중요하고, 지원대상·조건 설명은 표현이 다양하므로
#           두 방식을 결합"
#   - 하이브리드 가중치(BM25:kNN) 권장값 = 0.4 : 0.6 (탐색 범위 0.2~0.8)
#   - 검색 top-k 권장값 = 5 (LLM에 전달할 후보 문서 수)
#
# [핵심 설계 문제] BM25 점수와 kNN(코사인) 점수는 스케일이 완전히 다릅니다.
#   BM25는 이론상 상한이 없고 보통 0~20+ 사이, kNN(cosinesimil)은 대략 0~1~2 사이로 나옵니다.
#   두 점수를 그냥 0.4*bm25 + 0.6*knn으로 더하면 스케일이 큰 BM25가 거의 항상 이겨버려서
#   가중치를 아무리 조정해도 "가중합"의 의미가 없어집니다. 그래서 각 방식을 따로 검색해서 얻은
#   "후보군 내에서" min-max 정규화(0~1로 맞춤)를 먼저 하고, 그 정규화된 값끼리 가중합합니다.
#   이게 이 파일에서 가장 중요한 부분입니다 (normalize_scores 함수 참고).
#
# [필터 vs 검색어 - query_slots.py와의 연결]
#   query_slots.extract_slots()가 돌려주는 region/categories는 term/terms 필터로,
#   target_keywords는 BM25 검색어에 이어붙이는 힌트로 씁니다 (query_slots.py 상단 주석의
#   [설계 결정 1] 그대로 이어받음 - index_mapping.py에서 target이 "text" 타입이라 필터로 못 쓰기 때문).

import sys
from pathlib import Path

# query_slots.py와 동일한 이유로, src/indexing/*.py를 이 파일(src/rag/)에서 바로 import하기 위한
# 경로 설정입니다 (cwd에 의존하지 않도록 항상 __file__ 기준으로 계산 - fetch_bizinfo.py 교훈 참고).
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_INDEXING_DIR = _PROJECT_ROOT / "src" / "indexing"
if str(_INDEXING_DIR) not in sys.path:
    sys.path.insert(0, str(_INDEXING_DIR))

from config import INDEX_NAME, check_connection, get_client  # noqa: E402
from embedder import embed_batch  # noqa: E402  (질의 문장을 색인 때와 같은 모델/차원으로 벡터화하기 위해 재사용)

# ── 분석모델 정의서 1.3 "주요 파라미터"의 권장값을 그대로 상수화 ─────
BM25_WEIGHT = 0.4
KNN_WEIGHT = 0.6
DEFAULT_TOP_K = 5  # 최종적으로 LLM에 넘길 문서 수

# 가중합 재정렬을 하려면 top_k보다 넉넉한 후보 풀이 필요합니다 (예: kNN에서 3위였던 문서가
# BM25에서 순위 밖이었다면, BM25/kNN을 각각 top_k개씩만 가져오면 애초에 후보에 못 낌).
# top_k(5)의 4배 정도면 실제로 순위가 뒤바뀌는 경우를 대부분 커버하면서도 쿼리 비용은 작습니다.
CANDIDATE_POOL_SIZE = 20

# [실데이터로 발견한 문제 - 2026-09-21] "2026년 중앙부처 및 지자체 창업지원사업 통합 공고"처럼
# 수백 개 지원사업을 한 PDF에 몰아넣은 "통합 공고" 문서가 청크를 수십 개씩 만들어내는데, 이런
# 문서는 청크마다 전부 category="창업"이라 창업 관련 질의에 전부 높은 점수를 받습니다. top_k를
# 그냥 점수순으로 자르면 실제로 top-5가 "서로 다른 사업 5개"가 아니라 "같은 문서의 청크 5개"가
# 되어버리는 걸 실행해보고 확인함 - LLM에게 사실상 근거 문서 1개만 준 것과 같아지고, 평가 설계
# (분석모델 정의서 2.5)의 Hit@5/MRR도 "다른 사업이 정답인 질의"에서 깎입니다. 그래서 최종 top_k를
# 채울 때 같은 program_id는 이 값까지만 허용해서 강제로 다양성을 줍니다.
MAX_CHUNKS_PER_PROGRAM = 2

# 검색 결과 화면/LLM 컨텍스트에 필요한 필드만 가져옵니다. embedding(768개 숫자)은 응답 크기만
# 키우고 여기서는 안 쓰이므로 제외합니다.
_SOURCE_FIELDS = [
    "program_id", "program_name", "region_name", "category", "target",
    "apply_start", "apply_end", "apply_period_raw", "amount_hint",
    "chunk_text", "source_url",
]


# ── 1. 필터 절 구성 (filter_search_test.py의 build_bool_query()와 동일한 규칙) ──
def build_filter_clauses(region: str | None, categories: list[str] | None) -> list[dict]:
    filters = []
    if region:
        filters.append({"term": {"region_name": region}})
    if categories:
        filters.append({"terms": {"category": categories}})
    return filters


# ── 2. BM25 검색 ────────────────────────────────────────────────────
def search_bm25(client, query_text: str, target_keywords: list[str], filters: list[dict], size: int) -> dict:
    """
    출력: {chunk_id: {"score": BM25 점수, "source": _source 딕셔너리}} (chunk_id는 bulk_indexer.py에서
          _id로 chunk_id를 그대로 썼으므로 hit["_id"]가 곧 chunk_id입니다)
    """
    keyword_query = query_text
    if target_keywords:
        # query_slots.py의 [설계 결정 1]: target은 필터가 아니라 검색어 보강용
        keyword_query = f"{query_text} {' '.join(target_keywords)}"

    result = client.search(index=INDEX_NAME, body={
        "size": size,
        "query": {
            "bool": {
                "must": [{"match": {"chunk_text": keyword_query}}],
                "filter": filters,
            }
        },
        "_source": _SOURCE_FIELDS,
    })
    return {hit["_id"]: {"score": hit["_score"], "source": hit["_source"]} for hit in result["hits"]["hits"]}


# ── 3. kNN 검색 ─────────────────────────────────────────────────────
def search_knn(client, query_vector: list[float], filters: list[dict], size: int) -> dict:
    """출력: search_bm25()와 동일한 형태."""
    knn_query: dict = {"vector": query_vector, "k": size}
    if filters:
        # 필터가 2개 이상이면(예: region+category) bool로 묶어야 "둘 다 만족"으로 해석됩니다.
        # filter_search_test.py의 run_knn_with_filter()는 필터가 1개뿐이라 이 경우를 안 다뤘던 부분.
        knn_query["filter"] = {"bool": {"filter": filters}} if len(filters) > 1 else filters[0]

    result = client.search(index=INDEX_NAME, body={
        "size": size,
        "query": {"knn": {"embedding": knn_query}},
        "_source": _SOURCE_FIELDS,
    })
    return {hit["_id"]: {"score": hit["_score"], "source": hit["_source"]} for hit in result["hits"]["hits"]}


# ── 4. 점수 정규화 및 가중합 ─────────────────────────────────────────
def normalize_scores(scored: dict) -> dict[str, float]:
    """
    입력: {chunk_id: {"score": ..., "source": ...}}
    출력: {chunk_id: 0~1 사이로 정규화된 점수}

    min-max 정규화: 이 파일 상단 [핵심 설계 문제]에서 설명한 이유로, BM25/kNN 점수를 가중합하기 전에
    반드시 거쳐야 하는 단계입니다. "후보군 내에서" 최소~최대를 0~1로 늘리므로, 전체 인덱스 기준이
    아니라 이번 검색 결과 안에서의 상대적 순위를 보존하는 정규화입니다.
    """
    if not scored:
        return {}
    values = [v["score"] for v in scored.values()]
    lo, hi = min(values), max(values)
    if hi == lo:
        # 결과가 1건이거나 전부 동점이면 (hi-lo)가 0이 되어 0으로 나누기 에러가 납니다.
        # 이 경우 "다 똑같이 중요하다"는 뜻이므로 전부 만점(1.0)으로 취급합니다.
        return {chunk_id: 1.0 for chunk_id in scored}
    return {chunk_id: (v["score"] - lo) / (hi - lo) for chunk_id, v in scored.items()}


# ── 5. 공개 함수: 하이브리드 검색 ────────────────────────────────────
def hybrid_search(
    query_text: str,
    *,
    region: str | None = None,
    categories: list[str] | None = None,
    target_keywords: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    client=None,
) -> list[dict]:
    """
    입력: query_text - 사용자 원문 질의
          region/categories/target_keywords - query_slots.extract_slots()의 출력을 그대로 넣으면 됨
          top_k - 최종 반환 개수 (기본 5, 분석모델 정의서 권장값)
    출력: [{"chunk_id", "score", "bm25_score", "knn_score", **source필드}, ...]
          score 내림차순으로 정렬된 top_k개 (score는 0.4*BM25_norm + 0.6*KNN_norm)

    동작 순서:
      1) BM25/kNN을 각각 CANDIDATE_POOL_SIZE개씩 (필터 적용해서) 따로 검색
      2) 각각 min-max 정규화
      3) 두 결과의 합집합(둘 중 하나에만 있어도 포함)에 대해 가중합 점수 계산
         - 한쪽에만 있는 문서는 없는 쪽 점수를 0으로 취급 (예: kNN에는 없고 BM25에만 잡힌 문서는
           knn_norm=0으로 계산되므로 BM25_WEIGHT*bm25_norm 만큼만 점수를 받음 - 자연스럽게 페널티가 됨)
      4) score 기준 내림차순 정렬 후 top_k개 반환
    """
    client = client or get_client()
    if not check_connection(client):
        return []

    filters = build_filter_clauses(region, categories)

    bm25_hits = search_bm25(client, query_text, target_keywords or [], filters, CANDIDATE_POOL_SIZE)
    query_vector = embed_batch([query_text])[0]  # 색인 때와 같은 임베딩 모델(ko-sroberta)로 질의를 벡터화
    knn_hits = search_knn(client, query_vector, filters, CANDIDATE_POOL_SIZE)

    bm25_norm = normalize_scores(bm25_hits)
    knn_norm = normalize_scores(knn_hits)

    combined: list[dict] = []
    for chunk_id in set(bm25_hits) | set(knn_hits):
        bm25_score = bm25_norm.get(chunk_id, 0.0)
        knn_score = knn_norm.get(chunk_id, 0.0)
        source = (bm25_hits.get(chunk_id) or knn_hits.get(chunk_id))["source"]
        combined.append({
            "chunk_id": chunk_id,
            "score": BM25_WEIGHT * bm25_score + KNN_WEIGHT * knn_score,
            "bm25_score": bm25_hits.get(chunk_id, {}).get("score"),  # 원본(정규화 전) 점수 - 디버깅용
            "knn_score": knn_hits.get(chunk_id, {}).get("score"),
            **source,
        })

    combined.sort(key=lambda r: r["score"], reverse=True)

    # 다양성 확보: 점수순으로 훑으면서 같은 program_id는 MAX_CHUNKS_PER_PROGRAM개까지만 채택.
    # (위 MAX_CHUNKS_PER_PROGRAM 주석 참고 - 실제 "통합 공고" 문서로 확인된 문제에 대한 대응)
    selected: list[dict] = []
    count_by_program: dict[str, int] = {}
    for r in combined:
        program_id = r["program_id"]
        if count_by_program.get(program_id, 0) >= MAX_CHUNKS_PER_PROGRAM:
            continue
        selected.append(r)
        count_by_program[program_id] = count_by_program.get(program_id, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


# ── 수동 확인용 실행 블록 ────────────────────────────────────────────
if __name__ == "__main__":
    from query_slots import extract_slots, fetch_candidate_values  # noqa: E402  (같은 sys.path 설정 재사용)

    test_queries = [
        "서울 소상공인 창업 자금 지원사업 알려줘",
        "여성기업 대상 수출 지원",
    ]

    client = get_client()
    candidates = fetch_candidate_values(client)

    for q in test_queries:
        slots = extract_slots(q, candidates=candidates)
        print(f"[질의] {q}")
        print(f"  [슬롯] {slots}")
        results = hybrid_search(
            q,
            region=slots["region"],
            categories=slots["categories"],
            target_keywords=slots["target_keywords"],
            client=client,
        )
        for r in results:
            knn_display = f"{r['knn_score']:.3f}" if r["knn_score"] is not None else "None"
            print(f"    score={r['score']:.3f} (bm25={r['bm25_score']}, knn={knn_display}) "
                  f"[{r['category']}] {r['program_name'][:40]}")
        print()

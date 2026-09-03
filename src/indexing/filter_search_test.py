# [WBS 4.6] 지자체/분야 필터 조합 검색 테스트
#
# 실전 RAG 챗봇에서 사용자는 "서울에 있는 소상공인인데 창업 자금 지원받을 데 있어?"처럼
# 자유 문장으로 묻습니다. 이걸 그대로 kNN에만 넘기면 "의미는 비슷하지만 지역/분야가 안 맞는"
# 결과가 섞여 들어올 수 있습니다. 그래서 실전에서는 kNN/BM25 검색과 별개로
# region_name/category 같은 keyword 필드를 "필터"로 같이 걸어서 결과를 좁힙니다.
# 이 파일은 그 필터들이 실제로 의도대로 동작하는지 확인합니다.
#
# [2026-09-03 진행 전 확인된 것 - verify_index.py 결과]
#   region_name이 2,700건 전부 "서울"입니다 (크롤러 수집 범위를 서울로 좁혀뒀기 때문).
#   그래서 "지자체 필터"는 사실상 걸러낼 값이 하나뿐이라, ①~③은 category 조합 위주로 보고
#   region은 "문법이 동작하는지"만 확인하는 용도로 다룹니다.

import json

from config import INDEX_NAME, check_connection, get_client

# 실전 시나리오를 최대한 흉내낸 테스트 조합.
# (필터 없음 / 필터 1개 / 필터 2개 조합 / 존재하지 않는 값)을 섞어서,
# "필터가 있을 때"와 "없을 때"의 결과 차이가 실제로 나는지까지 확인합니다.
FILTER_CASES = [
    {"name": "필터 없음 (BM25만)", "region": None, "categories": None},
    {"name": "분야=창업 1개", "region": None, "categories": ["창업"]},
    {"name": "분야=창업 또는 금융 (복수)", "region": None, "categories": ["창업", "금융"]},
    {"name": "지역=서울 + 분야=창업 (조합)", "region": "서울", "categories": ["창업"]},
    {"name": "지역=경기 (실제로 없는 값 - 0건이 정상)", "region": "경기", "categories": None},
]

SEARCH_KEYWORD = "지원"  # chunk_text에 대한 BM25 검색어


# ── 1. bool 쿼리로 "검색 + 필터"를 함께 구성 ────────────────────────
def build_bool_query(keyword: str, region: str | None, categories: list[str] | None) -> dict:
    """
    입력: keyword(BM25로 찾을 단어), region(단일 값 또는 None), categories(여러 값 또는 None)
    출력: 오픈서치 bool 쿼리 딕셔너리

    [must vs filter - 이 테스트에서 가장 중요한 구분]
      must   : 점수 계산에 반영됨 (BM25 관련도 점수). "본문 검색"은 여기.
      filter : 점수 계산에 반영 안 되고 그냥 "있다/없다"로 통과시킴. "지역/분야 필터"는 여기.
      -> 필터를 must에 넣으면 "서울"이라는 단어가 본문에 많이 나올수록 점수가 올라가버려서,
         "지역이 서울인가"가 아니라 "서울이라는 단어가 본문에 자주 나오는가"로 왜곡됩니다.
         filter는 점수에 영향을 안 주고 순수하게 "거르는" 역할만 하므로 이게 맞는 자리입니다.

    region은 term(값 1개 정확히 일치), categories는 terms(여러 값 중 하나라도 일치)를 씁니다.
    "창업 또는 금융"처럼 OR 조건이 필요할 때 term을 여러 번 쓰는 대신 terms 하나로 표현합니다.
    """
    filters = []
    if region:
        filters.append({"term": {"region_name": region}})
    if categories:
        filters.append({"terms": {"category": categories}})

    return {
        "bool": {
            "must": [{"match": {"chunk_text": keyword}}],
            "filter": filters,
        }
    }


# ── 2. 필터 조합 테스트 실행 ────────────────────────────────────────
def run_filter_cases(client) -> None:
    print(f"── [1] 필터 조합 검색 (BM25 검색어: '{SEARCH_KEYWORD}') ──\n")
    for case in FILTER_CASES:
        query = build_bool_query(SEARCH_KEYWORD, case["region"], case["categories"])
        result = client.search(index=INDEX_NAME, body={
            "size": 3,
            "query": query,
            # 결과 검증을 눈으로 하기 쉽게, 필터 대상 필드 자체를 같이 받아옵니다.
            "_source": ["program_name", "region_name", "category"],
        })
        total = result["hits"]["total"]["value"]
        print(f"[{case['name']}] {total}건 매칭")
        for hit in result["hits"]["hits"][:2]:
            src = hit["_source"]
            print(f"    [{src['region_name']}/{src['category']}] {src['program_name'][:45]} (score={hit['_score']:.2f})")
        print()


# ── 3. kNN 벡터 검색 + 필터 조합 ────────────────────────────────────
def run_knn_with_filter(client) -> None:
    """
    BM25뿐 아니라 kNN(의미 기반) 검색도 필터를 같이 걸 수 있어야 실전에서 씁니다.
    오픈서치 kNN 쿼리는 knn.filter 자리에 term/bool 필터를 그대로 넣을 수 있습니다
    (필터를 먼저 적용한 뒤, 그 안에서만 벡터 검색을 하는 방식 - "pre-filtering").

    질문 문장은 매번 새로 임베딩해야 하는데, 이 프로젝트에서 임베딩 모델을 새로 불러오면
    무거우므로, 이미 색인된 문서 중 하나의 벡터를 "질문 대신" 재사용합니다.
    (검색 로직 자체를 검증하는 목적이라 실제 질문 문장이 아니어도 됩니다 - kNN+filter 문법이
    도는지가 핵심)
    """
    print("── [2] kNN 벡터 검색 + 필터 조합 ──\n")

    # "창업" 분야의 문서 하나를 골라서 그 벡터로 검색합니다.
    # -> 필터 없이 검색하면 다양한 분야가 섞여 나오지만, category=창업 필터를 걸면
    #    창업 분야 안에서만(의미상 가까운 순서로) 나와야 정상입니다.
    seed = client.search(index=INDEX_NAME, body={
        "size": 1,
        "query": {"term": {"category": "창업"}},
    })["hits"]["hits"][0]
    seed_vector = seed["_source"]["embedding"]
    print(f"[기준 문서] {seed['_source']['program_name'][:50]} (category=창업)\n")

    for label, filter_body in [
        ("필터 없음", None),
        ("category=창업 필터", {"term": {"category": "창업"}}),
        ("category=금융 필터 (기준 문서와 다른 분야)", {"term": {"category": "금융"}}),
    ]:
        knn_query: dict = {"vector": seed_vector, "k": 3}
        if filter_body:
            knn_query["filter"] = filter_body

        result = client.search(index=INDEX_NAME, body={
            "size": 3,
            "query": {"knn": {"embedding": knn_query}},
            "_source": ["program_name", "category"],
        })
        print(f"[{label}] {len(result['hits']['hits'])}건 반환")
        for hit in result["hits"]["hits"]:
            src = hit["_source"]
            print(f"    [{src['category']}] {src['program_name'][:45]} (score={hit['_score']:.3f})")
        print()


# ── 4. nori(한국어 형태소 분석기) 부재의 실제 영향 확인 ─────────────
def check_particle_gap(client) -> None:
    """
    index_mapping.py에 적어둔 [알려진 한계]를 실제 쿼리로 증명합니다.
    "지원사업" vs "지원사업을"처럼 조사만 붙은 단어가 서로 다른 결과 건수를 내면,
    지금 분석기(standard)가 조사를 못 떼어낸다는 뜻입니다. nori를 쓰면 두 검색이
    형태소 분석을 거쳐 "지원사업"이라는 같은 어근으로 정규화되어 결과가 같아집니다.
    """
    print("── [3] 한국어 조사 처리 한계 확인 (nori 미설치의 실제 영향) ──\n")
    pairs = [("지원사업", "지원사업을"), ("소상공인", "소상공인이")]
    for base, with_particle in pairs:
        count_base = client.count(index=INDEX_NAME, body={"query": {"match": {"chunk_text": base}}})["count"]
        count_particle = client.count(index=INDEX_NAME, body={"query": {"match": {"chunk_text": with_particle}}})["count"]
        same = count_base == count_particle
        print(f"  '{base}' {count_base}건  vs  '{with_particle}' {count_particle}건  "
              f"-> {'같음(문제 없음)' if same else '다름 (조사 미분리로 인한 결과 차이 확인됨)'}")
    print("\n  [해석] 결과가 다르게 나온다면: 사용자가 조사를 붙여 질문해도(예: '지원사업을 찾아줘')")
    print("  BM25 매칭 건수가 달라진다는 뜻입니다. kNN(의미 검색)은 조사에 영향을 안 받으므로")
    print("  지금처럼 BM25+kNN을 같이 쓰는 하이브리드 구조에서는 kNN이 이 약점을 상당 부분 보완합니다.")
    print()


def main() -> None:
    client = get_client()
    if not check_connection(client):
        return
    if not client.indices.exists(index=INDEX_NAME):
        print(f"[중단] 인덱스 '{INDEX_NAME}' 이(가) 없습니다.")
        return

    print(f"\n{'=' * 70}\n인덱스 '{INDEX_NAME}' 필터 조합 검색 테스트\n{'=' * 70}\n")
    run_filter_cases(client)
    run_knn_with_filter(client)
    check_particle_gap(client)


if __name__ == "__main__":
    main()

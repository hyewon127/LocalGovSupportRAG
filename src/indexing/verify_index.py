# [WBS 4.5] 색인 결과 검증 - 문서 수, 필드 매핑, 실제 검색 동작 확인
#
# 왜 별도 검증 단계가 필요한가:
#   Bulk 색인은 "성공 2,700건"이라고 찍혀도 그게 곧 "제대로 들어갔다"는 뜻은 아닙니다.
#   예를 들어 매핑이 의도와 달라 date가 text로 들어갔거나, embedding이 knn_vector가 아니라
#   float 배열로 들어갔으면, 색인은 성공하지만 나중에 날짜 필터/벡터 검색이 조용히 실패합니다.
#   전처리 단계에서 "확인용/정제 데이터 검증.py"로 청크를 검증했던 것과 같은 이유로,
#   색인 결과도 넘어가기 전에 숫자로 확인합니다.

import json

from config import CHUNKS_JSON, EMBEDDING_DIM, EMBEDDING_MODEL_NAME, INDEX_NAME, check_connection, get_client


# ── 1. 문서 수 검증 ─────────────────────────────────────────────────
def check_document_count(client) -> None:
    """
    원본 파일의 청크 수와 인덱스에 실제로 들어간 문서 수가 같은지 확인합니다.
    다르면 색인 중 일부가 실패했거나, 중복 _id로 덮어써진 것입니다.
    """
    indexed = client.count(index=INDEX_NAME)["count"]
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        source_count = len(json.load(f))

    print("── [1] 문서 수 ──")
    print(f"  원본 chunks.json : {source_count}건")
    print(f"  인덱스 색인됨    : {indexed}건")
    if indexed == source_count:
        print("  -> 일치 [OK]")
    else:
        print(f"  -> 불일치! 차이 {source_count - indexed}건 (색인 실패 또는 _id 중복 의심)")
    print()


# ── 2. 매핑 검증 ────────────────────────────────────────────────────
# index_mapping.py에서 "이렇게 되어야 한다"고 설계한 타입.
# 서버에 실제로 등록된 매핑과 대조해서, 의도대로 들어갔는지 자동으로 확인합니다.
EXPECTED_TYPES = {
    "chunk_id": "keyword",
    "program_id": "keyword",
    "program_name": "text",
    "region_name": "keyword",
    "category": "keyword",
    "apply_start": "date",
    "apply_end": "date",
    "apply_period_raw": "keyword",
    "target": "text",
    "amount_hint": "keyword",
    "chunk_text": "text",
    "token_count": "integer",
    "embedding": "knn_vector",
    "source_url": "keyword",
}


def check_mapping(client) -> None:
    """서버의 실제 매핑을 가져와서 EXPECTED_TYPES와 하나씩 대조합니다."""
    mapping = client.indices.get_mapping(index=INDEX_NAME)
    properties = mapping[INDEX_NAME]["mappings"]["properties"]

    print("── [2] 필드 매핑 ──")
    mismatches = []
    for field, expected in EXPECTED_TYPES.items():
        actual = properties.get(field, {}).get("type")
        if actual == expected:
            mark = "OK"
        else:
            mark = "XX"
            mismatches.append((field, expected, actual))
        print(f"  {mark} {field:18s} 기대={expected:12s} 실제={actual}")

    # 벡터 차원은 타입만 맞아서는 부족합니다 - 숫자가 모델과 같아야 검색이 동작합니다.
    actual_dim = properties.get("embedding", {}).get("dimension")
    # 표시 기호를 OK/XX 같은 영문으로 쓰는 이유: 윈도우 한국어 콘솔(cp949)은 체크표시(✓) 같은
    # 유니코드 기호를 출력하지 못해서 UnicodeEncodeError로 스크립트가 죽습니다(2026-09-03에 실제로 겪음).
    # 한글은 cp949에 있으므로 문제없고, 기호만 ASCII로 바꾸면 됩니다.
    print(f"  {'OK' if actual_dim == EMBEDDING_DIM else 'XX'} embedding.dimension 기대={EMBEDDING_DIM} 실제={actual_dim}")

    if mismatches or actual_dim != EMBEDDING_DIM:
        print("  -> 불일치가 있습니다. index_mapping.py의 create_index(recreate=True)로 재생성이 필요합니다.")
        print("     (오픈서치는 이미 만들어진 필드의 타입을 나중에 못 바꿉니다)")
    else:
        print("  -> 설계한 매핑과 완전히 일치 [OK]")
    print()


# ── 3. 필드 채움 상태 검증 ──────────────────────────────────────────
def check_field_coverage(client) -> None:
    """
    각 필드에 값이 실제로 들어있는 문서가 몇 건인지 셉니다.
    exists 쿼리 = "이 필드에 값이 있는 문서만" 찾는 쿼리 (null이면 없는 것으로 침).

    전처리 단계에서 확인했던 결측 현황(apply_start 1,445건 없음 등)이
    색인 후에도 그대로 유지되는지 대조하는 용도입니다. 숫자가 다르면
    색인 과정에서 값이 유실됐다는 뜻입니다.
    """
    print("── [3] 필드별 값 존재 건수 ──")
    total = client.count(index=INDEX_NAME)["count"]
    for field in ["chunk_text", "embedding", "region_name", "category",
                  "apply_start", "apply_end", "amount_hint", "source_url"]:
        body = {"query": {"exists": {"field": field}}}
        count = client.count(index=INDEX_NAME, body=body)["count"]
        print(f"  {field:18s} {count:5d} / {total} ({count / total:.0%})")
    print()


# ── 4. 실제 검색 동작 확인 ──────────────────────────────────────────
def check_search_works(client) -> None:
    """
    매핑이 맞아도 실제 쿼리가 도는지는 별개입니다.
    BM25(키워드) / term 필터 / kNN(벡터) 세 가지가 각각 동작하는지 한 번씩 눌러봅니다.
    """
    print("── [4] 검색 동작 확인 ──")

    # (1) BM25 키워드 검색: chunk_text 안에 '창업'이 들어간 문서
    bm25 = client.search(index=INDEX_NAME, body={
        "size": 1,
        "query": {"match": {"chunk_text": "창업"}},
    })
    hits = bm25["hits"]["total"]["value"]
    print(f"  BM25 '창업' 검색      : {hits}건 매칭")

    # (2) keyword 필터: region_name이 정확히 '서울'인 문서
    # keyword 타입이 제대로 잡혔는지 확인하는 용도 (text로 잘못 잡혔으면 term 검색이 0건이 나옵니다)
    term = client.search(index=INDEX_NAME, body={
        "size": 0,
        "query": {"term": {"region_name": "서울"}},
    })
    print(f"  term 필터 region=서울 : {term['hits']['total']['value']}건 매칭")

    # (3) kNN 벡터 검색: 아무 문서의 벡터를 하나 꺼내서, 그 벡터로 검색해봅니다.
    # 자기 자신이 1등으로 나와야 정상입니다(자기 자신과의 유사도가 가장 높으므로).
    # 이 테스트는 임베딩 모델을 다시 부를 필요가 없어서 빠르고, knn 검색 경로만 정확히 검증합니다.
    sample = client.search(index=INDEX_NAME, body={"size": 1, "query": {"match_all": {}}})
    sample_doc = sample["hits"]["hits"][0]
    sample_vector = sample_doc["_source"]["embedding"]

    knn = client.search(index=INDEX_NAME, body={
        "size": 3,
        "query": {"knn": {"embedding": {"vector": sample_vector, "k": 3}}},
        "_source": ["chunk_id", "program_name"],
    })
    top = knn["hits"]["hits"][0]
    is_self = top["_id"] == sample_doc["_id"]
    print(f"  kNN 벡터 검색         : {len(knn['hits']['hits'])}건 반환, "
          f"1등이 자기 자신인가 = {'예 [OK]' if is_self else '아니오 [XX]'}")
    print()


# ── 5. 샘플 문서 눈으로 확인 ────────────────────────────────────────
def show_sample_document(client) -> None:
    """실제로 들어간 문서 1건을 뽑아서 필드가 어떻게 저장됐는지 직접 봅니다."""
    result = client.search(index=INDEX_NAME, body={"size": 1, "query": {"match_all": {}}})
    source = result["hits"]["hits"][0]["_source"]

    print("── [5] 샘플 문서 1건 ──")
    for key, value in source.items():
        if key == "embedding":
            # 벡터는 768개 숫자라 다 찍으면 화면을 덮으므로 앞 3개만
            print(f"  {key:18s} [{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}, ...] (길이 {len(value)})")
        elif key == "chunk_text":
            print(f"  {key:18s} {str(value)[:60]}...")
        else:
            print(f"  {key:18s} {value}")
    print()


def main() -> None:
    client = get_client()
    if not check_connection(client):
        return
    if not client.indices.exists(index=INDEX_NAME):
        print(f"[중단] 인덱스 '{INDEX_NAME}' 이(가) 없습니다. index_mapping.py -> embedder.py -> bulk_indexer.py 순서로 실행하세요.")
        return

    print(f"\n{'=' * 70}")
    print(f"인덱스 '{INDEX_NAME}' 검증 (임베딩 모델: {EMBEDDING_MODEL_NAME}, {EMBEDDING_DIM}차원)")
    print(f"{'=' * 70}\n")

    check_document_count(client)
    check_mapping(client)
    check_field_coverage(client)
    check_search_works(client)
    show_sample_document(client)


if __name__ == "__main__":
    main()

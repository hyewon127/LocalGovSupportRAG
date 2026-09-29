# [WBS 6.3] /programs가 쓰는 OpenSearch 조회 로직 - 지원사업 목록/상세
#
# [핵심 문제] 인덱스(IDX_SUPPORT_CHUNK)의 문서 1건은 "사업 1건"이 아니라 "청크 1개"입니다.
#   테이블정의서 설계 메모대로 별도 사업 테이블 없이 사업 메타데이터를 청크마다 복사해 둔(비정규화) 구조라서,
#   그냥 검색하면 같은 사업이 청크 수만큼 반복해서 나옵니다 (통합 공고 하나가 청크 161개인 경우도 있음).
#   목록 화면에는 "사업"이 한 번씩만 나와야 하므로 아래 두 가지로 해결합니다.
#     - collapse(program_id): 검색 결과를 program_id별로 1건씩만 남김 (SQL의 GROUP BY와 비슷)
#     - cardinality(program_id): 조건에 맞는 "사업 수"를 따로 셈
#   collapse를 써도 hits.total은 여전히 "청크 수"로 나오는 걸 실행해서 확인했습니다 (창업 분야: 사업 41건인데
#   hits.total=361). 그래서 페이지 계산에 쓸 total은 반드시 cardinality 값을 써야 합니다.
#
# [메타데이터를 아무 청크에서나 가져와도 되는 이유] 같은 사업의 청크들은 category/apply_end/amount_hint가
#   전부 같은 값인지 491개 사업 전체를 집계로 확인했습니다 (불일치 0건 - chunker.py가 문서 단위로 추출해서
#   모든 청크에 복사하기 때문). 그래서 collapse가 고른 대표 청크 1개의 값을 사업 정보로 써도 됩니다.

from opensearchpy import OpenSearch

from config import INDEX_NAME  # src/indexing/config.py
# 접수기간 표기 규칙은 챗봇 답변 카드와 같아야 하므로 복사하지 않고 같은 함수를 씁니다 (WBS 8.4에서 중복 제거)
from context_builder import format_period  # src/rag/context_builder.py

# 목록/상세에서 필요한 필드만 가져옵니다. embedding(숫자 768개)을 빼지 않으면 응답이 수십 배로 커집니다.
_SUMMARY_FIELDS = [
    "program_id", "program_name", "region_name", "category", "target",
    "apply_start", "apply_end", "apply_period_raw", "amount_hint", "source_url",
]

# 한 사업의 청크를 전부 가져올 때 쓰는 상한. 실데이터 최대값은 161개(PBLN_000000000125563)라 여유 있게 잡음.
_MAX_CHUNKS_PER_PROGRAM_FETCH = 1000


def _to_summary(src: dict) -> dict:
    """청크 _source -> schemas.ProgramSummary 형태 (필드명을 화면 친화적으로 정리)."""
    return {
        "program_id": src["program_id"],
        "program_name": src["program_name"],
        "region_name": src["region_name"],
        "category": src.get("category"),
        "target": src.get("target"),
        "apply_start": src.get("apply_start"),
        "apply_end": src.get("apply_end"),
        "apply_period": format_period(src),
        "amount": src.get("amount_hint"),
        "source_url": src["source_url"],
    }


def _escape_wildcard(text: str) -> str:
    # wildcard 쿼리에서 *와 ?는 "아무 글자"라는 특수 의미라, 사용자가 입력한 *?는 글자 그대로 찾도록 이스케이프.
    # 백슬래시를 가장 먼저 바꿔야 뒤에서 추가한 백슬래시가 다시 이스케이프되지 않습니다.
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def list_programs(
    client: OpenSearch,
    *,
    region: str | None = None,
    category: str | None = None,
    q: str | None = None,
    page: int = 1,
    size: int = 20,
) -> tuple[int, list[dict]]:
    """
    출력: (조건에 맞는 사업 수, 해당 페이지의 사업 요약 리스트)

    정렬: 접수 마감일(apply_end) 늦은 순 -> 사업명 순.
      마감일이 날짜로 파싱되지 않은 사업(apply_end=None)은 missing="_last"로 맨 뒤로 보냅니다.
      이걸 안 하면 OpenSearch가 None을 어디에 둘지 기본 규칙대로 정해서 "상시 모집" 사업이 맨 앞을 차지할 수 있습니다.
    """
    filters = []
    if region:
        filters.append({"term": {"region_name": region}})
    if category:
        filters.append({"term": {"category": category}})
    if q:
        # [match가 아니라 wildcard를 쓰는 이유 - 실측 2026-09-22]
        #   index_mapping.py [알려진 한계]대로 nori(한국어 형태소 분석기)가 없어서 program_name은 공백 기준으로만
        #   쪼개져 있습니다. 그래서 match "창업"은 "창업중심대학", "창업경진대회" 같은 붙은 단어를 못 찾습니다.
        #   사업 수로 비교해보니 "창업" match 8건 vs wildcard 31건, "수출" 1건 vs 16건, "청년" 0건 vs 2건.
        #   사용자가 기대하는 건 "이 글자가 들어간 사업명"이라서 부분 문자열 검색(wildcard)이 맞습니다.
        #   앞에 *가 붙은 wildcard는 원래 느린 쿼리지만, 문서가 3,884건뿐이라 체감 차이가 없습니다.
        #   (nori를 설치해 인덱스를 다시 만들면 match로 되돌리는 게 정석입니다.)
        filters.append({"wildcard": {"program_name.keyword": {
            "value": f"*{_escape_wildcard(q)}*",
            "case_insensitive": True,  # "ai"로 검색해도 "AI 바우처" 사업이 나오게
        }}})

    body = {
        "from": (page - 1) * size,
        "size": size,
        "_source": _SUMMARY_FIELDS,
        # filter 절만 쓰는 이유: 목록은 "관련도 점수"로 줄 세울 필요가 없고(정렬 기준이 마감일), filter는
        # 점수 계산을 건너뛰고 결과를 캐시할 수 있어서 must보다 빠릅니다.
        "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
        "collapse": {"field": "program_id"},
        "sort": [
            {"apply_end": {"order": "desc", "missing": "_last"}},
            {"program_name.keyword": "asc"},
        ],
        "aggs": {"program_count": {"cardinality": {"field": "program_id"}}},
    }
    result = client.search(index=INDEX_NAME, body=body)
    total = result["aggregations"]["program_count"]["value"]
    items = [_to_summary(hit["_source"]) for hit in result["hits"]["hits"]]
    return total, items


def _chunk_seq(chunk_id: str) -> int:
    """
    "PBLN_000000000116904_10" -> 10

    [chunk_id로 바로 정렬하면 안 되는 이유] chunk_id는 keyword(문자열)라서 OpenSearch에서 정렬하면
    사전순이 되어 _0, _1, _10, _11, ..., _2 순서로 나옵니다 (실제로 조회해서 확인함). 원문 순서대로
    보여주려면 끝의 번호를 숫자로 바꿔서 정렬해야 합니다.
    """
    return int(chunk_id.rsplit("_", 1)[1])


def get_program(client: OpenSearch, program_id: str, max_chunks: int = 3) -> dict | None:
    """
    출력: schemas.ProgramDetail 형태의 dict, 없는 program_id면 None

    max_chunks: 본문 청크를 앞에서부터 몇 개까지 내려줄지. 화면정의서 SCR-02의 "원문 공고 요약" 영역에
      전부 싣기엔 큰 공고는 청크가 160개가 넘어서(응답 수백 KB), 기본값은 앞 3개(= 공고 앞부분)만 줍니다.
      전체 원문은 source_url(기업마당 공고 페이지)로 안내합니다.
    """
    result = client.search(index=INDEX_NAME, body={
        "size": _MAX_CHUNKS_PER_PROGRAM_FETCH,
        "_source": _SUMMARY_FIELDS + ["chunk_id", "chunk_text"],
        "query": {"term": {"program_id": program_id}},
    })
    hits = result["hits"]["hits"]
    if not hits:
        return None

    chunks = sorted((h["_source"] for h in hits), key=lambda s: _chunk_seq(s["chunk_id"]))
    return {
        **_to_summary(chunks[0]),
        "chunk_count": len(chunks),
        "chunks": [
            {"seq": _chunk_seq(s["chunk_id"]), "chunk_id": s["chunk_id"], "text": s["chunk_text"]}
            for s in chunks[:max_chunks]
        ],
    }

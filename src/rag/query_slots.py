# [WBS 5.1] 사용자 질의에서 검색 필터 슬롯 추출하기
#
# 배경: src/indexing/filter_search_test.py(WBS 4.6, 완료)의 build_bool_query(keyword, region, categories)는
# region/categories 값을 "이미 안다"고 가정하고 인자로 받았습니다(FILTER_CASES에 직접 값을 적어둠).
# 실제 챗봇에서는 사용자가 그 값을 미리 안 알려주고 "서울 소상공인인데 창업 자금 받을 데 있어?"처럼
# 자유 문장으로 묻습니다. 이 모듈은 그 자유 문장에서 region/categories(필터로 쓸 값)와
# target_keywords(필터는 아니지만 검색어에 보탤 힌트)를 뽑아내서, WBS 5.2(하이브리드 검색)가
# build_bool_query류 함수에 바로 넘길 수 있는 형태로 만들어줍니다.
#
# [설계 결정 1] region/category는 필터(filter)로, target은 검색어 힌트(keyword hint)로 분리했습니다.
#   index_mapping.py를 보면 region_name/category는 "keyword" 타입(정확히 일치해야 필터가 걸림)이고,
#   target은 "keyword" 서브필드가 없는 순수 "text" 타입(분석기가 쪼개서 BM25로만 매칭)입니다.
#   즉 target을 region/category처럼 term 필터로 쓰면 오픈서치 스펙상 애초에 "정확히 일치"가 거의
#   안 일어나서 필터링 의미가 없습니다 - 그래서 target에서 뽑은 키워드는 필터가 아니라
#   "chunk_text BM25 검색어에 보탤 힌트"로 다루도록 설계했습니다 (5.2에서 실제로 그렇게 씁니다).
#
# [설계 결정 2] region/category 후보값은 하드코딩된 스냅샷이 기본값이지만, 실제 인덱스에서 집계
#   (aggregation)로 최신값을 가져오는 fetch_candidate_values()도 같이 둡니다. 이유: region_name은
#   지금 "서울" 하나뿐인데, WBS10(지역 확장, 일정계획표 참고)에서 다른 지자체가 추가되면 하드코딩
#   상수는 코드를 고쳐야만 따라옵니다. 반면 아래 aggregation은 인덱스가 바뀌면 자동으로 최신값을
#   반환하므로, 운영 중인 서비스라면 이쪽이 더 안전합니다. (category는 기업마당 API가 고정한 8개
#   분류라 사실상 안 바뀌지만, 같은 함수로 같이 가져오는 게 코드가 더 단순합니다.)
#
# [설계 결정 3] 정규식(방식 A)과 LLM function calling(방식 B)을 하이브리드로 결합했습니다:
#   정규식이 뭔가(하나 이상)를 잡으면 그걸로 끝내고 LLM은 호출하지 않습니다 - 실제 질문 대부분은
#   "서울", "창업"처럼 후보값과 완전히 같은 단어를 그대로 포함하는 경우가 많아서, 정규식만으로
#   충분한 질문에까지 매번 API 비용/지연을 낼 이유가 없기 때문입니다. 정규식이 아무것도 못 잡았을 때만
#   (예: "요식업"처럼 후보 목록에 없는 동의어) LLM으로 한 번 더 시도합니다 - 확인용/질의 슬롯 추출.py에서
#   두 방식을 나란히 비교해본 결과를 바탕으로 내린 결정입니다.

import json
import re
import sys
from pathlib import Path

from openai import OpenAI

# LLM 접속 설정(업체/모델명)은 llm_client.py 한 곳에서 관리합니다 (분석모델 정의서 2.4 결정: Upstage).
from llm_client import LLM_MODEL, get_llm_client

# ── src/indexing/config.py 재사용을 위한 경로 설정 ───────────────────
# src/indexing/*.py들은 전부 "from config import ..."라는 같은 폴더 기준 import를 쓰는데,
# 이 파일(src/rag/query_slots.py)은 다른 폴더에 있어서 그대로는 import가 안 됩니다.
# cwd에 의존하는 방식(예: sys.path.append("../indexing"))은 fetch_bizinfo.py가 겪었던 것과 같은
# "실행 위치에 따라 결과가 달라지는" 버그를 또 만들 수 있으므로, 항상 __file__ 기준 절대경로로 계산합니다.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # src/rag/ -> src/ -> 프로젝트 루트
_INDEXING_DIR = _PROJECT_ROOT / "src" / "indexing"
if str(_INDEXING_DIR) not in sys.path:
    sys.path.insert(0, str(_INDEXING_DIR))

from config import INDEX_NAME, check_connection, get_client  # noqa: E402 (경로 설정 뒤에 와야 하는 import)


# ── 후보값 스냅샷 (2026-09-21, chunks.json 3,884건 기준 실데이터로 확인) ──
# fetch_candidate_values()로 최신값을 못 가져올 때(OpenSearch가 꺼져 있을 때 등) 쓰는 대체값입니다.
REGION_VALUES_SNAPSHOT = ["서울"]
CATEGORY_VALUES_SNAPSHOT = ["기술", "경영", "인력", "수출", "창업", "금융", "내수", "기타"]

# target은 index_mapping.py상 "text" 타입(키워드 집계 불가)이라 실인덱스에서 자동으로 못 뽑아옵니다.
# metadata.csv 원문을 직접 훑어서 자주 보이는 값들을 사람이 추린 목록이라 완전하지 않습니다 - 그래서
# 필터가 아니라 "검색어 힌트"로만 쓰는 게 이 목록의 불완전함을 감안해도 안전한 이유이기도 합니다.
TARGET_KEYWORDS_SNAPSHOT = ["소상공인", "중소기업", "창업벤처", "여성기업", "장애인기업", "사회적기업", "제조업"]


def fetch_candidate_values(client=None) -> dict:
    """
    입력: OpenSearch 클라이언트 (없으면 새로 생성)
    출력: {"region": [...], "category": [...]} - 인덱스에 실제로 존재하는 값만 집계(aggregation)로 반환
          (인덱스에 접속 못 하면 위 스냅샷 상수로 대체하고 경고를 출력함)

    terms aggregation은 SQL의 "SELECT DISTINCT region_name, COUNT(*) ... GROUP BY region_name"과
    비슷합니다 - keyword 필드라서 가능한 집계입니다 (target처럼 text 필드는 이 방식으로 안 됩니다).
    """
    if client is None:
        client = get_client()

    if not check_connection(client) or not client.indices.exists(index=INDEX_NAME):
        print("[경고] 인덱스에 연결할 수 없어 후보값을 하드코딩 스냅샷으로 대체합니다.")
        return {"region": REGION_VALUES_SNAPSHOT, "category": CATEGORY_VALUES_SNAPSHOT}

    result = client.search(index=INDEX_NAME, body={
        "size": 0,  # 문서 본문은 필요 없고 집계 결과만 필요하므로 0건 (응답 속도/용량 절약)
        "aggs": {
            "regions": {"terms": {"field": "region_name", "size": 50}},
            "categories": {"terms": {"field": "category", "size": 50}},
        },
    })
    regions = [bucket["key"] for bucket in result["aggregations"]["regions"]["buckets"]]
    categories = [bucket["key"] for bucket in result["aggregations"]["categories"]["buckets"]]
    return {"region": regions, "category": categories}


# ── 방식 A: 정규식/키워드 매칭 (빠름·무료·결정적) ────────────────────
def _extract_slots_regex(query: str, region_values: list[str], category_values: list[str]) -> dict:
    """
    입력: 사용자 자유 질의, region/category 후보값 리스트
    출력: {"region": str|None, "categories": list[str], "target_keywords": list[str]}
    동작: 후보값이 질의 문자열에 그대로(부분 문자열로) 포함되는지만 확인합니다.
          re.escape를 쓰는 이유: 후보값에 향후 "."이나 "?" 같은 정규식 특수문자가 섞여 들어와도
          (예: 지역명에 "." 포함 등) 정규식 문법으로 오해되지 않고 항상 "글자 그대로"로 매칭되게 하기 위함입니다.
    """
    region = None
    for r in region_values:
        if re.search(re.escape(r), query):
            region = r
            break  # region_name은 스키마상 값 1개짜리 필드이므로(build_chunk_record 참고) 첫 매칭에서 멈춤

    categories = [c for c in category_values if re.search(re.escape(c), query)]
    target_keywords = [t for t in TARGET_KEYWORDS_SNAPSHOT if re.search(re.escape(t), query)]

    return {"region": region, "categories": categories, "target_keywords": target_keywords}


# ── 방식 B: LLM function calling (동의어/암시적 표현에 강함, API 비용 발생) ──
def _build_slot_function_schema(region_values: list[str], category_values: list[str]) -> dict:
    """
    region/categories는 "실제로 존재하는 값만" enum으로 못 박아둡니다. 모델이 enum 밖의 값을
    지어내면(예: 실제 없는 지역명) build_bool_query()의 term/terms 필터에 넣었을 때 무조건
    0건이 나오는 문제가 생기기 때문입니다. target_keywords는 필터가 아니라 검색어 힌트일 뿐이라
    enum으로 제한하지 않고, 모델이 질의의 의미를 보고 자유롭게 뽑아내게 둡니다.
    """
    return {
        "type": "function",
        "function": {
            "name": "extract_search_slots",
            "description": "사용자의 지원사업 검색 질의에서 필터/검색에 쓸 슬롯 값을 추출한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": f"질의에 언급된 지자체명. 다음 값만 가능: {region_values}. "
                                        f"언급이 없으면 빈 문자열(\"\").",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string", "enum": category_values},
                        "description": f"질의 의미상 해당하는 지원분야(복수 가능, 이 값들 중에서만 선택: "
                                        f"{category_values}). 명확히 해당하는 게 없으면 빈 배열.",
                    },
                    "target_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "질의에서 언급되거나 암시된 지원대상/업종 키워드(복수 가능, 자유 형식). "
                                        "필터가 아니라 본문 검색어 보강용이므로 정해진 목록에 없어도 된다. "
                                        "없으면 빈 배열.",
                    },
                },
                "required": ["region", "categories", "target_keywords"],
            },
        },
    }


def _extract_slots_llm(query: str, client: OpenAI, region_values: list[str], category_values: list[str]) -> dict:
    """
    입력: 사용자 자유 질의, LLM 클라이언트(llm_client.get_llm_client()), region/category 후보값
    출력: _extract_slots_regex()와 동일한 형태의 dict
    동작: tool_choice="required"로 "반드시 도구(함수)를 호출해라"라고 강제합니다.
          (강제하지 않으면 모델이 함수 호출 대신 그냥 일반 텍스트로 답해버릴 수 있음 - function calling에서
          자주 나는 실수라 명시적으로 막아둠)
          특정 함수 이름을 지정하는 {"type":"function","function":{"name":...}} 형식 대신 "required"를
          쓴 이유: 도구가 1개뿐이라 의미는 완전히 같고, "required"는 OpenAI 호환 API(Upstage 등)들이
          더 널리 지원하는 형식이라 업체를 바꿔도 안전합니다.
    """
    schema = _build_slot_function_schema(region_values, category_values)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "너는 지자체 지원사업 검색 챗봇의 질의 분석기다."},
            {"role": "user", "content": query},
        ],
        tools=[schema],
        tool_choice="required",
        temperature=0,  # 슬롯 추출은 "정답이 하나"인 분류 작업이라 답변 생성(0.2)보다 더 결정적으로 둠
    )
    tool_call = response.choices[0].message.tool_calls[0]
    raw = json.loads(tool_call.function.arguments)  # "이 인자로 호출하세요"라는 JSON 문자열 -> dict로 변환
    raw["region"] = raw.get("region") or None  # 빈 문자열은 "없음"과 같은 의미이므로 None으로 통일
    return raw


# ── 공개 함수: 정규식 우선, 실패 시 LLM 폴백 ─────────────────────────
def extract_slots(
    query: str,
    *,
    candidates: dict | None = None,
    llm_client: OpenAI | None = None,
) -> dict:
    """
    입력: query - 사용자 자유 질의
          candidates - {"region": [...], "category": [...]} (없으면 하드코딩 스냅샷 사용.
                       fetch_candidate_values()로 얻은 최신값을 넘기는 걸 권장)
          llm_client - LLM 클라이언트 (없으면 llm_client.get_llm_client()로 자동 생성 시도.
                       키가 없으면 LLM 폴백을 건너뛰고 정규식 결과만 반환)
    출력: {"region": str|None, "categories": list[str], "target_keywords": list[str]}

    동작: 정규식(방식 A)을 먼저 시도 -> region/categories/target_keywords 중 하나라도 잡혔으면
          그 결과를 그대로 반환(API 비용 절약). 셋 다 비어 있을 때만 LLM(방식 B)으로 한 번 더 시도.
    """
    candidates = candidates or {"region": REGION_VALUES_SNAPSHOT, "category": CATEGORY_VALUES_SNAPSHOT}
    region_values = candidates["region"]
    category_values = candidates["category"]

    slots = _extract_slots_regex(query, region_values, category_values)
    if slots["region"] or slots["categories"] or slots["target_keywords"]:
        return slots

    client = llm_client if llm_client is not None else get_llm_client()
    if client is None:
        return slots  # LLM을 쓸 수 없으면 (정규식이 아무것도 못 찾은) 빈 결과라도 그대로 반환

    try:
        return _extract_slots_llm(query, client, region_values, category_values)
    except Exception as e:
        # 슬롯 추출 실패가 검색 자체를 막으면 안 되므로(필터 없이도 BM25/kNN은 여전히 동작함),
        # 예외를 여기서 흡수하고 정규식 결과(빈 값)로 안전하게 폴백합니다.
        print(f"[경고] LLM 슬롯 추출 실패, 필터 없이 진행합니다: {e}")
        return slots


# ── 수동 확인용 실행 블록 (index_mapping.py의 show_mapping() 등과 같은 패턴) ──
if __name__ == "__main__":
    test_queries = [
        "서울에 있는 소상공인인데 창업 자금 지원받을 데 있어?",
        "여성기업 대상 수출 지원사업 알려줘",
        "요식업 하는데 받을 수 있는 지원금 있나요",
        "인력 채용하면 지원금 나오는 사업",
    ]

    candidates = fetch_candidate_values()
    print(f"[후보값] region={candidates['region']} / category={candidates['category']}\n")

    for q in test_queries:
        print(f"[질의] {q}")
        print(f"  -> {extract_slots(q, candidates=candidates)}\n")

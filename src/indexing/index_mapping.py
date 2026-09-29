# [WBS 4.1] 인덱스 매핑 설계 - IDX_SUPPORT_CHUNK의 필드 타입 정의 및 인덱스 생성
#
# "매핑(mapping)"이란: 관계형 DB의 테이블 스키마(CREATE TABLE)에 해당하는 것으로,
# "이 인덱스에는 어떤 필드가 있고, 각 필드를 어떤 방식으로 저장/검색할 것인가"를 미리 정의하는 것.
# 오픈서치는 매핑 없이도 문서를 넣으면 타입을 자동 추론(dynamic mapping)해주지만,
# 자동 추론에 맡기면 "2026-08-14"를 date가 아니라 text로 잡아버리는 식으로 어긋날 수 있어서
# 검색/필터가 의도대로 안 됩니다. 그래서 넣기 전에 직접 정의합니다.
#
# [이 파일에서 내린 설계 결정 3가지 - 2026-09-03]
#   결정1) 인덱스 이름은 소문자 idx_support_chunk (오픈서치가 대문자 인덱스명을 거부함) -> config.py 참고
#   결정2) knn 엔진은 faiss (아래 KNN_ENGINE 주석 참고 - nmslib은 이 버전에서 제거됨)
#   결정3) 한국어 형태소 분석기(nori) 미설치 상태이므로 chunk_text는 기본 분석기로 색인
#          -> BM25 한글 검색 품질에 한계가 있음. 아래 [알려진 한계] 참고

import json

from config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME, INDEX_NAME, check_connection, get_client

# ── 1. kNN(벡터 검색) 엔진 설정 ─────────────────────────────────────
# 오픈서치에서 벡터 검색을 담당하는 라이브러리를 "엔진"이라고 부르며, 3가지가 있었습니다.
#   - nmslib : 예전 기본값. OpenSearch 3.x에서 제거됨 -> 지금 쓰면 인덱스 생성이 HTTP 400으로 실패
#              (확인용/오픈서치 임베딩 색인 맛보기.py는 nmslib로 짜여 있어서 지금 그대로 돌리면 실패함)
#   - faiss  : 페이스북(Meta)이 만든 벡터 검색 라이브러리. 대용량/성능에 유리. 현재 버전에서 정상 동작 확인
#   - lucene : 오픈서치 내장 엔진. 별도 네이티브 라이브러리 없이 동작. 소규모에 무난
# 2026-09-03에 실제 서버(OpenSearch 3.8.0)에 인덱스를 만들어보며 확인한 결과 faiss/lucene은 200,
# nmslib은 400이었습니다. 2,700건 규모면 셋 다 무방하지만, 나중에 데이터가 늘어날 걸 감안해 faiss 선택.
KNN_ENGINE = "faiss"

# space_type: "두 벡터가 얼마나 가까운가"를 재는 방식
#   cosinesimil = 코사인 유사도. 벡터의 "방향"만 보고 "길이"는 무시함.
#   문장 임베딩은 문장이 길수록 벡터 크기가 커지는 경향이 있는데, 우리가 알고 싶은 건
#   "길이"가 아니라 "의미가 비슷한 방향인가"이므로 코사인이 적합합니다.
KNN_SPACE_TYPE = "cosinesimil"


# ── 2. 인덱스 매핑 정의 ─────────────────────────────────────────────
def build_index_body() -> dict:
    """
    입력: 없음 (config.py의 EMBEDDING_DIM을 참조)
    출력: 오픈서치 인덱스 생성 API에 그대로 넘길 딕셔너리 (settings + mappings)

    [text vs keyword - 이 프로젝트에서 가장 중요한 타입 구분]
      keyword : 값을 쪼개지 않고 통째로 저장. "서울"로 정확히 일치하는 것만 찾음.
                필터링(where절 같은 것)/집계에 적합. 부분 검색은 안 됨.
      text    : 분석기가 단어 단위로 쪼개서 저장. "청년 창업 지원"으로 검색하면
                "청년", "창업", "지원"이 들어간 문서를 BM25 점수순으로 찾아줌.
      -> "지자체=서울인 것만 걸러내기"는 keyword가 맞고,
         "본문에서 청년창업 관련 내용 찾기"는 text가 맞습니다.
    """
    return {
        "settings": {
            "index": {
                # knn: true -> 이 인덱스에서 knn_vector 필드로 벡터 검색을 하겠다는 선언.
                # 이걸 빼면 knn_vector 필드를 정의해도 검색 시 에러가 납니다.
                "knn": True,
                # 로컬 PC에 노드가 1개뿐이므로 복제본(replica)은 0으로 둡니다.
                # 기본값 1로 두면 복제본을 할당할 다른 노드가 없어서 클러스터 상태가 계속 yellow로 남습니다.
                "number_of_replicas": 0,
                # 샤드(데이터를 나눠 담는 단위)도 2,700건 규모에서는 1개면 충분합니다.
                # 오히려 여러 개로 쪼개면 각 샤드의 BM25 통계가 나뉘어 점수가 살짝 왜곡될 수 있습니다.
                "number_of_shards": 1,
            }
        },
        "mappings": {
            "properties": {
                # ── 식별자 (검색 대상이 아니라 "정확히 일치"로만 쓰이는 값들) ──
                "chunk_id": {"type": "keyword"},    # 청크 1개의 고유 ID (문서 _id로도 씀)
                "program_id": {"type": "keyword"},  # 사업 1개의 ID. "같은 사업의 청크 모두 보기" 필터에 사용
                # ── 사업명: 두 가지 용도가 다 필요한 대표적인 필드 ──
                # program_name       -> "청년창업"으로 부분 검색(BM25)
                # program_name.keyword -> "2026년 서울 청년창업 지원사업"과 정확히 일치하는 것만 필터/집계
                # fields로 서브필드를 두면 한 필드를 두 방식으로 동시에 색인할 수 있습니다 (multi-field).
                "program_name": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                # ── 필터 조건으로 쓰일 값들 (WBS 4.6 "지자체/분야 필터 조합 검색"의 대상) ──
                "region_name": {"type": "keyword"},  # 예: "서울" - 정확히 일치로 거름
                "category": {"type": "keyword"},     # 예: "창업", "금융" - 정확히 일치로 거름
                # ── 신청 기간 ──
                # format을 명시하는 이유: 안 적으면 오픈서치가 여러 날짜 형식을 추측하는데,
                # 우리 데이터는 chunker.py의 parse_period()가 "YYYY-MM-DD"로만 만들어주므로 고정합니다.
                # 값이 null인 청크가 1,445건(54%) 있지만, 오픈서치는 null을 그냥 "색인 안 함"으로
                # 처리하므로 에러가 나지 않습니다 ("접수중인 사업만" 필터에서 자동으로 빠질 뿐).
                "apply_start": {"type": "date", "format": "yyyy-MM-dd"},
                "apply_end": {"type": "date", "format": "yyyy-MM-dd"},
                # 원문 그대로의 접수기간 문자열("예산 소진시까지" 등). 날짜로 못 바꾼 값을 버리지 않고
                # 보존하는 필드이므로, 날짜 타입이 아니라 keyword로 둡니다 (그대로 화면에 보여주는 용도).
                "apply_period_raw": {"type": "keyword"},
                # ── 지원 대상/금액 ──
                "target": {"type": "text"},       # "중소기업", "만 19~39세 청년" 등 문장형이라 text
                "amount_hint": {"type": "keyword"},  # "5,000만원" 같은 정규식 추출값. 계산이 아니라 표시용이라 keyword
                # ── 본문 (BM25 키워드 검색의 실제 대상) ──
                "chunk_text": {"type": "text"},
                "token_count": {"type": "integer"},  # 청크 크기 분석/디버깅용 숫자 필드
                # ── 임베딩 벡터 (kNN 의미 검색의 대상) ──
                # dimension은 임베딩 모델의 출력 크기와 반드시 같아야 합니다.
                # 다르면 색인할 때 "Vector dimension mismatch" 에러가 납니다.
                # 그래서 config.py의 EMBEDDING_DIM을 직접 참조해서, 모델을 바꾸면 여기도 자동으로 따라오게 했습니다.
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",  # 근사 최근접 이웃(ANN) 탐색 알고리즘. 전수 비교보다 훨씬 빠름
                        "space_type": KNN_SPACE_TYPE,
                        "engine": KNN_ENGINE,
                    },
                },
                "source_url": {"type": "keyword"},  # 원문 링크. 검색 대상이 아니라 답변에 근거로 붙일 값
            }
        },
    }


# ── 3. 인덱스 생성 ──────────────────────────────────────────────────
def create_index(recreate: bool = False) -> bool:
    """
    입력: recreate - True면 기존 인덱스를 삭제하고 새로 만듦 (데이터가 날아감!)
                     False면 이미 있을 때 건드리지 않고 그냥 둠 (기본값, 안전)
    출력: 인덱스가 사용 가능한 상태면 True

    recreate 기본값을 False로 둔 이유: 실수로 이 스크립트를 다시 실행했을 때
    이미 색인해둔 문서(2026-09-22 기준 3,884건)가 통째로 날아가면 안 되기 때문입니다.
    매핑을 바꿔서 다시 만들어야 할 때만 명시적으로 recreate=True를 줍니다.
    (오픈서치는 이미 만들어진 필드의 타입을 나중에 바꿀 수 없어서, 매핑 변경 = 인덱스 재생성입니다)
    """
    client = get_client()
    if not check_connection(client):
        return False

    exists = client.indices.exists(index=INDEX_NAME)

    if exists and not recreate:
        print(f"[건너뜀] 인덱스 '{INDEX_NAME}' 이(가) 이미 존재합니다. (매핑을 바꾸려면 recreate=True)")
        return True

    if exists and recreate:
        client.indices.delete(index=INDEX_NAME)
        print(f"[삭제] 기존 인덱스 '{INDEX_NAME}' 을(를) 삭제했습니다 (recreate=True)")

    body = build_index_body()
    client.indices.create(index=INDEX_NAME, body=body)
    print(f"[생성 완료] 인덱스 '{INDEX_NAME}'")
    print(f"           임베딩 모델: {EMBEDDING_MODEL_NAME} / 차원: {EMBEDDING_DIM}")
    print(f"           kNN 엔진: {KNN_ENGINE} / 유사도: {KNN_SPACE_TYPE}")
    return True


def show_mapping() -> None:
    """서버에 실제로 등록된 매핑을 그대로 출력합니다 (내가 의도한 대로 들어갔는지 눈으로 확인용)."""
    client = get_client()
    if not check_connection(client):
        return
    if not client.indices.exists(index=INDEX_NAME):
        print(f"[없음] 인덱스 '{INDEX_NAME}' 이(가) 아직 없습니다.")
        return
    mapping = client.indices.get_mapping(index=INDEX_NAME)
    print(json.dumps(mapping, ensure_ascii=False, indent=2))


# [알려진 한계 - 2026-09-03]
# 이 서버에는 한국어 형태소 분석기 플러그인(analysis-nori)이 설치돼 있지 않습니다.
# 그래서 chunk_text는 기본(standard) 분석기로 쪼개지는데, 기본 분석기는 한글을 공백 기준으로만
# 나누기 때문에 "지원사업을"과 "지원사업이"를 서로 다른 단어로 봅니다(조사가 안 떨어짐).
# -> BM25 키워드 검색의 재현율이 떨어질 수 있습니다.
# 해결하려면: bin/opensearch-plugin install analysis-nori 로 플러그인을 설치한 뒤,
#             chunk_text에 "analyzer": "nori" 를 지정하고 인덱스를 재생성해야 합니다.
# 지금은 WBS 5(하이브리드 검색)에서 kNN 벡터 검색이 이 약점을 상당 부분 보완해주므로,
# 우선 이대로 진행하고 검색 품질을 본 뒤에 판단하는 것으로 미뤄둡니다.


if __name__ == "__main__":
    # 기본 실행은 "없으면 만들고, 있으면 그대로 두기" (안전한 쪽)
    # 매핑을 바꾼 뒤 다시 만들고 싶으면 아래 줄을 create_index(recreate=True)로 바꿔서 실행하세요.
    create_index(recreate=False)

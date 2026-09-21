# [WBS 4.4] Bulk API 배치 색인 스크립트 - chunks_embedded.json을 오픈서치에 넣기
#
# 입력: data/processed/chunks_embedded.json (embedder.py 출력, embedding이 채워진 2,700개)
# 출력: 오픈서치 인덱스 idx_support_chunk 에 문서 2,700개 색인
#
# [Bulk API가 뭐고 왜 쓰는가]
#   맛보기 스크립트에서 썼던 client.index()는 문서 1개당 HTTP 요청 1번입니다.
#   2,700개면 요청 2,700번 -> 매 요청마다 접속/응답 대기가 붙어서 매우 느립니다.
#   Bulk API는 "문서 여러 개를 한 요청에 몰아서" 보냅니다. 요청 횟수가 수십 번으로 줄어
#   보통 수십 배 빨라집니다. 대량 색인에서는 사실상 필수입니다.

import json

from opensearchpy import helpers  # bulk() 헬퍼 함수가 여기 들어있음

from config import EMBEDDED_JSON, EMBEDDING_DIM, INDEX_NAME, check_connection, get_client

# 한 번의 Bulk 요청에 몇 개 문서를 담을지.
# 768차원 벡터가 문서마다 붙어서 1건이 대략 10~20KB입니다. 200개면 요청 1건이 2~4MB 정도로,
# 오픈서치 기본 요청 크기 제한(100MB)에 한참 못 미치면서도 충분히 효율적인 크기입니다.
BULK_CHUNK_SIZE = 200


# ── 1. 색인할 문서를 하나씩 만들어내는 제너레이터 ───────────────────
def generate_actions(records: list[dict]):
    """
    입력: chunks_embedded.json에서 읽은 레코드 리스트
    출력: (yield로) 오픈서치 bulk가 이해하는 형태의 딕셔너리를 하나씩 내보냄

    왜 리스트가 아니라 제너레이터(yield)인가:
      2,700개 문서 전체를 또 하나의 리스트로 만들면 메모리에 데이터가 두 벌 올라갑니다.
      제너레이터는 "필요할 때 하나씩" 만들어주므로 helpers.bulk()가 배치 단위로 가져가면서
      메모리를 크게 안 씁니다. 데이터가 더 커져도 그대로 동작합니다.

    _id를 chunk_id로 지정하는 이유(중요):
      _id를 안 주면 오픈서치가 임의의 ID를 만들어 붙입니다. 그러면 이 스크립트를 두 번 돌렸을 때
      같은 내용이 2,700개 더 들어가서 총 5,400개가 됩니다(중복).
      _id를 chunk_id로 고정하면 같은 ID는 "덮어쓰기"가 되므로, 몇 번을 다시 돌려도
      항상 2,700개를 유지합니다 (이런 성질을 멱등성/idempotent 하다고 합니다).
    """
    for record in records:
        yield {
            "_index": INDEX_NAME,
            "_id": record["chunk_id"],
            "_source": record,  # chunks_embedded.json의 필드 구조가 이미 매핑과 1:1이라 그대로 넣음
        }


# ── 2. 색인 전 데이터 점검 ──────────────────────────────────────────
def validate_records(records: list[dict]) -> list[dict]:
    """
    입력: 원본 레코드 리스트
    출력: 색인해도 되는 레코드만 걸러낸 리스트

    여기서 미리 거르는 이유: Bulk는 문서 몇 개가 잘못돼도 나머지는 그냥 들어갑니다.
    그래서 "일부만 실패"한 상태를 나중에 발견하기 어렵습니다. 넣기 전에 확인해서
    몇 건이 왜 빠졌는지 먼저 알고 시작하는 편이 낫습니다.
    """
    valid = []
    no_embedding = 0
    wrong_dim = 0

    for record in records:
        embedding = record.get("embedding")
        if not embedding:
            # embedder.py를 안 돌렸거나 중간에 실패한 청크
            no_embedding += 1
            continue
        if len(embedding) != EMBEDDING_DIM:
            # 모델을 바꿨는데 임베딩을 다시 안 만든 경우 - 넣어봤자 색인 에러가 납니다
            wrong_dim += 1
            continue
        valid.append(record)

    print(f"[점검] 전체 {len(records)}건 중 색인 대상 {len(valid)}건")
    if no_embedding:
        print(f"       제외 - 임베딩 없음: {no_embedding}건 (embedder.py를 먼저 실행하세요)")
    if wrong_dim:
        print(f"       제외 - 차원 불일치: {wrong_dim}건 (현재 설정 {EMBEDDING_DIM}차원)")
    return valid


# ── 3. Bulk 색인 실행 ───────────────────────────────────────────────
def run_bulk_index() -> bool:
    client = get_client()
    if not check_connection(client):
        return False

    # 인덱스가 없으면 색인 시 오픈서치가 자동으로 만들어버리는데(dynamic mapping),
    # 그러면 우리가 index_mapping.py에서 설계한 타입(knn_vector 등)이 아니라
    # 자동 추론된 엉뚱한 타입으로 만들어집니다. 그래서 반드시 먼저 확인합니다.
    if not client.indices.exists(index=INDEX_NAME):
        print(f"[중단] 인덱스 '{INDEX_NAME}' 이(가) 없습니다. index_mapping.py를 먼저 실행하세요.")
        return False

    if not EMBEDDED_JSON.exists():
        print(f"[중단] {EMBEDDED_JSON} 이(가) 없습니다. embedder.py를 먼저 실행하세요.")
        return False

    with open(EMBEDDED_JSON, encoding="utf-8") as f:
        records = json.load(f)

    valid_records = validate_records(records)
    if not valid_records:
        return False

    print(f"[색인 시작] {len(valid_records)}건을 {BULK_CHUNK_SIZE}개씩 나눠서 전송합니다...")

    # helpers.bulk(): 제너레이터에서 문서를 꺼내 배치로 묶어 보내주고, 결과를 집계해줍니다.
    #   raise_on_error=False -> 문서 몇 개가 실패해도 예외를 던지지 않고 끝까지 진행한 뒤
    #                            실패 목록을 돌려줍니다. 2,700건 중 3건 실패했다고 전체가
    #                            중단되면 오히려 곤란하므로, 끝까지 넣고 실패분만 따로 봅니다.
    success_count, errors = helpers.bulk(
        client,
        generate_actions(valid_records),
        chunk_size=BULK_CHUNK_SIZE,
        raise_on_error=False,
        request_timeout=120,  # 벡터가 커서 한 배치 전송에 시간이 걸릴 수 있음
    )

    # refresh: 오픈서치는 색인 직후 바로 검색되지 않고 기본 1초 주기로 반영됩니다.
    # 바로 다음 단계(verify_index.py)에서 문서 수를 세야 하므로 강제로 반영시킵니다.
    client.indices.refresh(index=INDEX_NAME)

    print(f"[색인 완료] 성공 {success_count}건 / 실패 {len(errors)}건")
    if errors:
        # 실패 사유는 보통 몇 가지로 반복되므로 앞의 3건만 보여주면 원인 파악에 충분합니다.
        print("[실패 사례 - 최대 3건]")
        for err in errors[:3]:
            print(f"  {json.dumps(err, ensure_ascii=False)[:300]}")
    return len(errors) == 0


if __name__ == "__main__":
    run_bulk_index()

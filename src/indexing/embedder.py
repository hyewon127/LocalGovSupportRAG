# [WBS 4.3] 임베딩 생성 스크립트 - chunks.json(2,700개)의 chunk_text를 벡터로 변환
#
# 입력: data/processed/chunks.json          (chunker.py 출력, embedding 필드가 전부 None)
# 출력: data/processed/chunks_embedded.json (embedding 필드가 숫자 배열로 채워진 같은 구조)
#
# 색인(④)과 임베딩(③)을 왜 분리했는가:
#   임베딩은 느리고(로컬 모델이면 수 분), 색인은 빠릅니다. 한 파일에 합쳐두면
#   "색인 설정을 조금 고쳐서 다시 돌리고 싶을 때"마다 임베딩을 처음부터 다시 계산해야 합니다.
#   중간 결과를 파일로 떨궈두면 임베딩은 한 번만 하고 색인만 몇 번이든 다시 할 수 있습니다.
#   (extractor.py -> chunker.py 로 단계마다 JSON을 남겼던 것과 같은 방식)

import json
import time

from config import (
    CHUNKS_JSON,
    EMBEDDED_JSON,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
    PROCESSED_DIR,
)

# 한 번에 몇 개씩 묶어서 임베딩할지.
# 너무 작으면(1) 호출 오버헤드 때문에 느리고, 너무 크면 메모리를 많이 쓰거나 API 요청 크기 제한에 걸립니다.
BATCH_SIZE = 32

_model_cache = None  # 모델을 한 번만 로딩해서 재사용하기 위한 캐시 (매 배치마다 새로 로딩하면 몇 배로 느려짐)


# ── 1. 모델별 임베딩 함수 ───────────────────────────────────────────
def _embed_batch_ko_sroberta(texts: list[str]) -> list[list[float]]:
    """로컬 ko-sroberta 모델로 배치 임베딩 (API 호출 없음, 내 PC의 CPU/GPU를 씀)."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        print(f"[모델 로딩] {EMBEDDING_MODEL_NAME} (최초 1회는 다운로드 때문에 시간이 걸립니다)")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # encode()는 numpy 배열을 돌려주는데, JSON으로 저장하려면 파이썬 list여야 합니다.
    return [vec.tolist() for vec in _model_cache.encode(texts)]


def _embed_batch_openai(texts: list[str]) -> list[list[float]]:
    """OpenAI 임베딩 API로 배치 임베딩 (유료, .env의 OPENAI_API_KEY 필요)."""
    global _model_cache
    if _model_cache is None:
        from openai import OpenAI
        _model_cache = OpenAI()  # 여기서는 '모델'이 아니라 API 클라이언트를 캐시
    response = _model_cache.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
    return [item.embedding for item in response.data]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    입력: 문자열 리스트 (최대 BATCH_SIZE개)
    출력: 각 문자열의 임베딩 벡터 리스트

    config.py의 EMBEDDING_PROVIDER 값에 따라 어느 함수를 쓸지 결정합니다.
    이렇게 한 군데서 갈라주면, 모델을 바꿀 때 이 파일의 나머지 코드는 하나도 안 고쳐도 됩니다.
    """
    if EMBEDDING_PROVIDER == "ko-sroberta":
        return _embed_batch_ko_sroberta(texts)
    if EMBEDDING_PROVIDER == "openai":
        return _embed_batch_openai(texts)
    raise ValueError(f"알 수 없는 EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


# ── 2. 이어하기(resume) 지원 ────────────────────────────────────────
def load_existing_embeddings() -> dict[str, list[float]]:
    """
    입력: 없음 (EMBEDDED_JSON 파일이 있으면 읽음)
    출력: {chunk_id: embedding 벡터} 딕셔너리. 파일이 없으면 빈 딕셔너리.

    왜 필요한가: 2,700개 임베딩은 로컬 모델 기준 몇 분씩 걸립니다. 중간에 껐거나 에러가 나면
    처음부터 다시 하는 건 낭비이므로, 이미 만들어둔 벡터는 재사용합니다.
    (청크 내용이 바뀌면 chunk_id도 바뀌므로, 낡은 벡터가 잘못 재사용될 걱정은 없습니다)
    """
    if not EMBEDDED_JSON.exists():
        return {}
    with open(EMBEDDED_JSON, encoding="utf-8") as f:
        previous = json.load(f)
    # 벡터가 실제로 들어있는 것만 재사용 (embedding이 None인 레코드는 아직 안 된 것)
    done = {r["chunk_id"]: r["embedding"] for r in previous if r.get("embedding")}
    if done:
        # 차원이 다르면(모델을 바꾼 경우) 재사용하면 안 됩니다 - 섞이면 색인 때 터집니다.
        sample_dim = len(next(iter(done.values())))
        if sample_dim != EMBEDDING_DIM:
            print(f"[무시] 기존 파일의 벡터 차원({sample_dim})이 현재 설정({EMBEDDING_DIM})과 달라 재사용하지 않습니다.")
            print("       (임베딩 모델을 바꾼 경우 - 전부 새로 계산합니다)")
            return {}
    return done


# ── 3. 전체 임베딩 ──────────────────────────────────────────────────
def embed_all() -> None:
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        chunks = json.load(f)

    already_done = load_existing_embeddings()
    todo = [c for c in chunks if c["chunk_id"] not in already_done]

    print(f"[대상] 전체 청크 {len(chunks)}개")
    print(f"       이미 임베딩됨: {len(already_done)}개 / 이번에 처리: {len(todo)}개")
    print(f"[모델] {EMBEDDING_MODEL_NAME} ({EMBEDDING_PROVIDER}, {EMBEDDING_DIM}차원)")

    start = time.time()
    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start:batch_start + BATCH_SIZE]
        vectors = embed_batch([c["chunk_text"] for c in batch])

        # 첫 배치에서 차원을 검증합니다. 여기서 걸러내지 않으면 2,700개를 다 계산한 뒤
        # 색인 단계에서야 "차원 불일치" 에러를 만나게 되어 시간을 통째로 날립니다.
        if batch_start == 0 and len(vectors[0]) != EMBEDDING_DIM:
            raise ValueError(
                f"모델이 만든 벡터 차원({len(vectors[0])})이 config.py의 EMBEDDING_DIM({EMBEDDING_DIM})과 다릅니다. "
                "config.py의 _EMBEDDING_SPECS를 실제 모델에 맞게 고치세요."
            )

        for chunk, vector in zip(batch, vectors):
            already_done[chunk["chunk_id"]] = vector

        done_count = min(batch_start + BATCH_SIZE, len(todo))
        elapsed = time.time() - start
        # 진행률 표시: 오래 걸리는 작업에서 "멈춘 건지 도는 건지" 알 수 있어야 하기 때문
        print(f"\r  진행 {done_count}/{len(todo)} ({done_count / len(todo):.0%}) "
              f"경과 {elapsed:.0f}초", end="", flush=True)

    print()  # 진행률 줄바꿈

    # 원본 chunks.json의 구조를 그대로 유지한 채 embedding 필드만 채웁니다.
    # (스키마를 바꾸지 않아야 색인 단계에서 그대로 넣을 수 있음)
    for chunk in chunks:
        chunk["embedding"] = already_done.get(chunk["chunk_id"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDED_JSON, "w", encoding="utf-8") as f:
        # indent 없이 저장: 2,700 x 768개 숫자라서 들여쓰기를 넣으면 파일이 수십 MB 더 커집니다.
        # (사람이 직접 읽을 파일이 아니라 색인 스크립트가 읽을 파일이므로 가독성보다 용량이 우선)
        json.dump(chunks, f, ensure_ascii=False)

    embedded_count = sum(1 for c in chunks if c["embedding"])
    print(f"[완료] {embedded_count}/{len(chunks)}개 임베딩 -> {EMBEDDED_JSON}")
    print(f"       총 소요 {time.time() - start:.0f}초")


if __name__ == "__main__":
    embed_all()

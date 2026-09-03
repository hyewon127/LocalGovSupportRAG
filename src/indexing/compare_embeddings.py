# [WBS 4.2] 임베딩 모델 소규모 비교 테스트 - text-embedding-3-small vs ko-sroberta
#
# 왜 이 단계가 필요한가:
#   인덱스 매핑의 knn_vector dimension은 모델을 정해야 확정됩니다(1536 vs 768).
#   그리고 오픈서치는 한 번 만든 필드의 타입/차원을 나중에 못 바꾸기 때문에,
#   "일단 아무거나로 색인해보고 나중에 바꾸지 뭐"가 안 됩니다(인덱스를 통째로 다시 만들어야 함).
#   그래서 2,700건 전체를 색인하기 "전에" 소규모로 두 모델을 비교해서 먼저 정합니다.
#
# 무엇을 비교하는가 (라벨링된 정답 데이터가 없으므로, 아래 3가지를 봅니다):
#   1) 검색 결과의 질  : 실제 질문 몇 개를 던져서 각 모델이 어떤 청크를 1등으로 뽑는지 눈으로 비교
#   2) 변별력(중요)    : 상위 점수와 전체 평균 점수의 차이. 모든 문서에 0.99를 주는 모델은
#                        "다 비슷하다"고 말하는 것과 같아서 검색에 쓸모가 없습니다.
#                        1등 점수가 평균보다 뚜렷하게 높아야 "잘 골라내는" 모델입니다.
#   3) 비용/속도       : API 호출 비용과 소요 시간 (ko-sroberta는 무료지만 내 PC 자원을 씀)

import json
import os
import time

from config import CHUNKS_JSON  # config.py를 import하는 시점에 load_dotenv()가 실행되어 .env 값이 환경변수로 올라옴

# ── 1. 비교 조건 ────────────────────────────────────────────────────
SAMPLE_SIZE = 40  # 비교에 쓸 청크 개수. 너무 적으면 우연히 결과가 뒤집히고, 너무 많으면 비용/시간이 늘어남

# 실제 사용자가 챗봇에 물어볼 법한 질문들. "본문에 그 단어가 그대로 없는" 질문을 일부러 섞었습니다.
# (단어가 그대로 있으면 BM25로도 찾을 수 있어서, 임베딩의 진짜 실력인 "의미 검색"을 못 봅니다)
TEST_QUERIES = [
    "청년이 창업할 때 받을 수 있는 지원금이 있나요?",
    "소상공인 대상 자금 지원 사업 알려줘",
    "수출하려는 중소기업을 도와주는 제도",
    "교육이나 컨설팅을 무료로 받을 수 있는 프로그램",
]


# ── 2. 코사인 유사도 계산 ───────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    입력: 벡터 2개 (같은 길이의 float 리스트)
    출력: -1 ~ 1 사이의 유사도 (1에 가까울수록 의미가 비슷함)

    공식: (a·b) / (|a| * |b|)  = 내적을 각 벡터의 길이로 나눔
    -> 벡터의 "길이"는 나눠서 없애고 "방향"만 비교하겠다는 뜻.
       문장이 길수록 벡터가 커지는 효과를 제거하기 위함입니다.
    numpy 없이 순수 파이썬으로 쓴 이유: 이 계산이 실제로 뭘 하는지 코드로 드러내기 위함
    (실전 검색은 오픈서치가 알아서 해주므로 성능은 중요하지 않은 자리입니다)
    """
    dot = sum(x * y for x, y in zip(a, b))          # 내적: 같은 위치 원소끼리 곱해서 전부 더함
    norm_a = sum(x * x for x in a) ** 0.5            # 벡터 a의 길이(유클리드 norm)
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:                   # 0벡터가 들어오면 0으로 나누게 되므로 방어
        return 0.0
    return dot / (norm_a * norm_b)


# ── 3. 모델별 임베딩 함수 ───────────────────────────────────────────
def embed_openai(texts: list[str]) -> list[list[float]]:
    """
    입력: 문자열 리스트
    출력: 각 문자열의 임베딩 벡터 리스트 (1536차원)

    OpenAI 임베딩 API는 여러 문장을 리스트로 한 번에 보낼 수 있습니다(배치).
    하나씩 40번 호출하는 것보다 훨씬 빠르고, 요금은 토큰 수 기준이라 동일합니다.
    """
    from openai import OpenAI  # 함수 안에서 import: 이 모델을 안 쓸 때는 불러오지 않기 위함

    client = OpenAI()  # API 키는 .env의 OPENAI_API_KEY를 자동으로 읽어감
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    # 응답의 data는 입력 순서와 동일한 순서로 돌아옵니다 (index 필드로도 확인 가능)
    return [item.embedding for item in response.data]


def embed_ko_sroberta(texts: list[str]) -> list[list[float]]:
    """
    입력: 문자열 리스트
    출력: 각 문자열의 임베딩 벡터 리스트 (768차원)

    이 모델은 API가 아니라 내 PC에서 직접 돌아갑니다(무료).
    대신 sentence-transformers 라이브러리와 모델 파일(약 440MB) 다운로드가 필요합니다.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return [v.tolist() for v in model.encode(texts)]  # numpy 배열 -> 파이썬 list로 변환


def is_ko_sroberta_available() -> bool:
    """sentence-transformers가 설치돼 있는지 확인 (없으면 이 모델 비교는 건너뜀)."""
    try:
        import sentence_transformers  # noqa: F401  (설치 여부만 확인하는 용도라 실제로 안 씀)
        return True
    except ImportError:
        return False


def is_openai_available() -> bool:
    """
    .env에 "쓸 수 있는" OPENAI_API_KEY가 들어있는지 확인.

    단순히 값이 있는지만 보면 안 되는 이유: .env 템플릿에 처음 적혀 있던 "your_key_here" 같은
    placeholder도 "값이 있는 것"으로 잡히기 때문입니다. 실제로 2026-09-03에 이 상태로 실행해서
    401 AuthenticationError를 맞았고, 그때 스크립트 전체가 죽어서 ko-sroberta 결과도 못 봤습니다.
    -> 그래서 명백한 placeholder는 미리 걸러내고, 그래도 통과하면 실제 호출에서 잡습니다.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return False
    # OpenAI 키는 "sk-"로 시작합니다. 이 형태가 아니면 placeholder로 간주.
    return key.startswith("sk-")


# ── 4. 한 모델에 대해 검색을 돌려보고 결과를 정리 ───────────────────
def evaluate_model(name: str, embed_fn, chunk_texts: list[str], chunk_labels: list[str]) -> dict:
    """
    입력: name       - 화면에 표시할 모델 이름
          embed_fn   - 위에서 정의한 임베딩 함수 (embed_openai / embed_ko_sroberta)
          chunk_texts - 검색 대상이 될 청크 본문 40개
          chunk_labels - 각 청크가 어떤 사업인지 알아보기 위한 사업명 40개
    출력: {"name":..., "dim":..., "elapsed":..., "results":[질문별 결과...]}

    동작: 청크 40개를 벡터로 만들고, 질문도 벡터로 만든 다음,
          질문 벡터와 가장 가까운 청크를 코사인 유사도로 정렬해서 상위 3개를 뽑습니다.
          이게 나중에 오픈서치의 kNN 검색이 하는 일과 정확히 같은 계산입니다
          (오픈서치는 이걸 훨씬 빠르게 근사 계산으로 해줄 뿐).
    """
    print(f"\n{'=' * 70}\n[{name}] 임베딩 생성 중...")
    start = time.time()
    chunk_vectors = embed_fn(chunk_texts)             # 청크 40개를 한 번에 임베딩
    query_vectors = embed_fn(TEST_QUERIES)            # 질문 4개도 임베딩
    elapsed = time.time() - start

    dim = len(chunk_vectors[0])
    print(f"[{name}] 완료 - 차원={dim}, 소요시간={elapsed:.1f}초")

    results = []
    for query, q_vec in zip(TEST_QUERIES, query_vectors):
        # 모든 청크와의 유사도를 계산 -> [(유사도, 청크번호), ...]
        scored = [(cosine_similarity(q_vec, c_vec), i) for i, c_vec in enumerate(chunk_vectors)]
        scored.sort(reverse=True)  # 유사도가 높은 순으로 정렬

        top_score = scored[0][0]
        avg_score = sum(s for s, _ in scored) / len(scored)
        results.append({
            "query": query,
            "top3": [(score, chunk_labels[i]) for score, i in scored[:3]],
            "top_score": top_score,
            "avg_score": avg_score,
            # 변별력 = 1등 점수 - 전체 평균. 클수록 "이게 정답이다"라고 확실히 말해주는 모델
            "discrimination": top_score - avg_score,
        })

    return {"name": name, "dim": dim, "elapsed": elapsed, "results": results}


# ── 5. 비교 결과 출력 ───────────────────────────────────────────────
def print_report(reports: list[dict]) -> None:
    """두 모델의 결과를 질문별로 나란히 출력해서 눈으로 비교할 수 있게 함."""
    for i, query in enumerate(TEST_QUERIES):
        print(f"\n{'=' * 70}")
        print(f"[질문 {i + 1}] {query}")
        for report in reports:
            r = report["results"][i]
            print(f"\n  ── {report['name']} (변별력 {r['discrimination']:.3f} "
                  f"= 1등 {r['top_score']:.3f} - 평균 {r['avg_score']:.3f})")
            for rank, (score, label) in enumerate(r["top3"], start=1):
                print(f"     {rank}위 [{score:.3f}] {label[:55]}")

    print(f"\n{'=' * 70}\n[종합]")
    for report in reports:
        # 질문 4개의 변별력 평균 - 이 값이 높은 모델이 "정답과 오답을 더 잘 구분"함
        avg_disc = sum(r["discrimination"] for r in report["results"]) / len(report["results"])
        print(f"  {report['name']:28s} 차원={report['dim']:5d}  "
              f"평균 변별력={avg_disc:.3f}  소요={report['elapsed']:.1f}초")


def main() -> None:
    # 비교용 샘플 추출: 앞에서부터 40개가 아니라 전체에서 고르게 뽑습니다.
    # 앞에서부터 자르면 같은 사업의 청크만 잔뜩 뽑혀서(한 사업이 청크 10개 이상) 비교가 무의미해지기 때문.
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        all_chunks = json.load(f)

    step = len(all_chunks) // SAMPLE_SIZE          # 전체를 SAMPLE_SIZE등분 해서
    sample = all_chunks[::step][:SAMPLE_SIZE]      # 일정 간격으로 골라 뽑음 -> 여러 사업이 고루 섞임
    chunk_texts = [c["chunk_text"] for c in sample]
    chunk_labels = [c["program_name"] for c in sample]

    print(f"[비교 대상] 청크 {len(sample)}개 (전체 {len(all_chunks)}개 중 균등 간격 추출)")
    print(f"[테스트 질문] {len(TEST_QUERIES)}개")

    # 두 모델 중 "지금 이 PC에서 실제로 돌릴 수 있는 것"만 평가합니다.
    # 한쪽이 준비 안 됐다고 스크립트 전체가 죽으면, 준비된 쪽 결과조차 못 보기 때문입니다.
    reports = []

    if is_openai_available():
        reports.append(evaluate_model("OpenAI text-embedding-3-small", embed_openai, chunk_texts, chunk_labels))
    else:
        print("\n[건너뜀] OpenAI - .env의 OPENAI_API_KEY가 없거나 placeholder입니다(sk-로 시작해야 함).")

    if is_ko_sroberta_available():
        reports.append(evaluate_model("ko-sroberta-multitask(로컬)", embed_ko_sroberta, chunk_texts, chunk_labels))
    else:
        print("\n[건너뜀] ko-sroberta - sentence-transformers 미설치 (pip install sentence-transformers)")

    if not reports:
        print("\n[중단] 비교할 수 있는 모델이 하나도 없습니다. 위 안내대로 준비한 뒤 다시 실행하세요.")
        return

    print_report(reports)

    if len(reports) == 1:
        # 한 모델만 돌았을 때 "비교했다"고 착각하지 않도록 명시적으로 알려줍니다.
        print(f"\n[주의] 실제로 평가된 모델은 '{reports[0]['name']}' 하나뿐입니다.")
        print("       WBS 4.2의 원래 취지(두 모델 비교)를 채우려면 나머지 한쪽도 준비해서 다시 실행하세요.")


if __name__ == "__main__":
    main()

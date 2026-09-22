# [WBS 4 공통] 오픈서치 색인 단계에서 4개 스크립트가 공유하는 설정 모음
#
# 왜 별도 파일로 뺐는가:
#   index_mapping.py(①) / embedder.py(③) / bulk_indexer.py(④) / verify_index.py(⑤) 가
#   전부 "같은 인덱스 이름"과 "같은 임베딩 차원"을 알아야 합니다. 이걸 각 파일에 복사해두면
#   나중에 임베딩 모델을 바꿀 때 4군데를 다 고쳐야 하고, 한 군데라도 빠뜨리면
#   "매핑은 1536차원인데 넣는 벡터는 768차원" 같은 색인 에러가 납니다.
#   그래서 chunker.py에서 청킹 파라미터를 맨 위에 모아뒀던 것과 같은 이유로, 여기 한 곳에만 둡니다.

import os
from pathlib import Path

from dotenv import load_dotenv          # .env 파일의 값을 os.environ으로 읽어오는 함수
from opensearchpy import OpenSearch     # 오픈서치 서버에 HTTP로 요청을 보내는 클라이언트 라이브러리

load_dotenv()  # 프로젝트 루트의 .env를 읽어서 환경변수로 올림 (API 키/호스트 정보를 코드에 안 박기 위함)


# ── 1. 경로 설정 ────────────────────────────────────────────────────
# chunker.py와 동일하게 "이 파일 위치" 기준으로 고정합니다.
# (cwd 기준 상대경로를 쓰면 어느 폴더에서 실행하느냐에 따라 경로가 깨짐 - fetch_bizinfo.py에서 겪었던 버그)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent          # src/indexing/ -> src/ -> 프로젝트 루트
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSON = PROCESSED_DIR / "chunks.json"            # chunker.py의 출력 (청크 3,884개, embedding=None 상태)
EMBEDDED_JSON = PROCESSED_DIR / "chunks_embedded.json"  # embedder.py(③)의 출력 (embedding 채워진 상태)


# ── 2. 오픈서치 접속 설정 ───────────────────────────────────────────
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

# 인덱스 이름: 테이블정의서에는 "IDX_SUPPORT_CHUNK"(대문자)로 적혀 있지만,
# 오픈서치는 인덱스 이름에 대문자를 허용하지 않습니다(생성 시 400 에러).
# 그래서 실제 인덱스 이름은 소문자로 쓰고, 테이블정의서의 이름과 1:1로 대응된다고 이해하면 됩니다.
INDEX_NAME = "idx_support_chunk"


# ── 3. 임베딩 모델 설정 ─────────────────────────────────────────────
# 여기가 "모델을 바꾸는 스위치"입니다. EMBEDDING_PROVIDER만 바꾸면
# 차원(EMBEDDING_DIM)도 같이 따라 바뀌도록 묶어놨습니다 - 둘을 따로 관리하면
# "모델은 바꿨는데 차원은 안 바꿔서" 생기는 색인 에러가 반드시 나기 때문입니다.
#
#   "openai"      : text-embedding-3-small (1536차원) - API 호출, 유료(아주 저렴), 계획서상 정식 스택
#   "ko-sroberta" : jhgan/ko-sroberta-multitask (768차원) - 내 PC에서 로컬 실행, 무료, 첫 실행 시 모델 다운로드
#
# [결정 2026-09-03] ko-sroberta 선택.
#   이유: .env에 유효한 OPENAI_API_KEY가 없는 상태였고, 학습/포트폴리오 목적상 과금 없이
#   전체 파이프라인을 끝까지 돌려보는 게 우선이라고 판단. 한국어 특화 모델이라 국문 공고문
#   임베딩 품질도 나쁘지 않음.
#   [영향] 테이블정의서에는 embedding 차원이 1536(OpenAI 기준)으로 적혀 있으므로,
#   768로 바뀐 것을 테이블정의서에도 반영해야 문서-코드가 어긋나지 않습니다.
#   나중에 OpenAI로 바꾸려면: 이 값을 "openai"로 바꾸고 -> 키 설정 -> 인덱스 재생성
#   (create_index(recreate=True)) -> 임베딩 재생성 -> 재색인 순서로 진행해야 합니다.
#   차원이 다른 벡터는 같은 인덱스에 못 들어가기 때문에 인덱스 재생성이 필수입니다.
EMBEDDING_PROVIDER = "ko-sroberta"

_EMBEDDING_SPECS = {
    "openai": {
        "model_name": "text-embedding-3-small",
        "dim": 1536,
    },
    "ko-sroberta": {
        "model_name": "jhgan/ko-sroberta-multitask",
        "dim": 768,
    },
}

EMBEDDING_MODEL_NAME = _EMBEDDING_SPECS[EMBEDDING_PROVIDER]["model_name"]
EMBEDDING_DIM = _EMBEDDING_SPECS[EMBEDDING_PROVIDER]["dim"]


# ── 4. 오픈서치 클라이언트 생성 ─────────────────────────────────────
def get_client() -> OpenSearch:
    """
    입력: 없음 (위의 HOST/PORT 설정을 사용)
    출력: OpenSearch 클라이언트 객체

    주의: 이 객체를 만든다고 해서 서버에 접속이 되는 건 아닙니다.
    "요청을 보낼 준비"만 하는 것이고, 실제 연결은 client.info() 같은 요청을 보낼 때 일어납니다.
    그래서 서버가 꺼져 있어도 이 함수 자체는 에러 없이 통과하고, 나중에 요청할 때 터집니다.
    -> 각 스크립트가 시작할 때 check_connection()으로 먼저 확인하는 이유입니다.
    """
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        use_ssl=False,       # 이 프로젝트의 오픈서치는 zip 설치 + 보안 플러그인 비활성화 -> HTTP 평문 통신
        verify_certs=False,  # HTTPS 환경(도커 기본 설정 등)이라면 True + http_auth 설정이 추가로 필요
        http_compress=True,  # 응답을 gzip으로 압축해서 받음 (2,700건 색인처럼 데이터가 클 때 유리)
        timeout=30,          # Bulk 색인은 한 번에 오래 걸릴 수 있어서 맛보기(10초)보다 넉넉하게 잡음
    )


def check_connection(client: OpenSearch) -> bool:
    """
    입력: OpenSearch 클라이언트
    출력: 접속 성공하면 True, 실패하면 False (실패 사유를 콘솔에 출력)

    각 스크립트 맨 앞에서 호출해서, "서버가 안 떠 있는데 한참 작업하다가 마지막에 터지는" 일을 막습니다.
    """
    try:
        info = client.info()
        print(f"[연결 성공] OpenSearch {info['version']['number']} @ {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
        return True
    except Exception as e:
        print(f"[연결 실패] {OPENSEARCH_HOST}:{OPENSEARCH_PORT} 에 접속할 수 없습니다: {e}")
        print("           OpenSearch가 켜져 있는지 확인하세요 (bin\\opensearch.bat 실행)")
        return False

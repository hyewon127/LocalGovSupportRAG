# [WBS 6.1] FastAPI 앱 진입점 - 서버 시작/종료 처리, 공통 예외 처리, 라우터 등록
#
# 실행 방법 (프로젝트 루트에서, OpenSearch가 먼저 떠 있어야 함):
#   .venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
#   -> http://127.0.0.1:8000/docs 에서 API 문서(Swagger UI)를 보고 바로 호출해볼 수 있습니다.
#   포트 8000은 분석모델 정의서 1.2 "포트번호: 8000 (FastAPI)"를 따른 것입니다.
#   서버 준비까지 약 20초 걸립니다 (대부분 임베딩 모델 로딩 - 아래 lifespan 참고).
#
#   [주의 - localhost 말고 127.0.0.1로 부를 것, 2026-09-22 실측]
#   uvicorn은 기본적으로 IPv4(127.0.0.1)에만 연결을 받습니다. 그런데 Windows에서 "localhost"는 IPv6(::1)를
#   먼저 시도하고, 거절당한 뒤 IPv4로 넘어가기까지 약 2초를 기다립니다. 같은 /health 요청이
#   127.0.0.1로는 14~30ms, localhost로는 2,058~2,069ms 걸렸습니다 - 응답시간 목표(5초)의 40%를 주소 해석에만 씀.
#   그래서 Streamlit(WBS 6.4)의 API 주소 기본값도 127.0.0.1로 둡니다.
#
# [폴더 구조] 분석모델 정의서 1.1 시스템 구조도의 "FastAPI 서버 (RAG 오케스트레이션)" 칸을 아래처럼 나눴습니다.
#   main.py            - 앱 생성, 서버 시작 시 자원 준비(lifespan), 공통 예외 처리, /health
#   schemas.py         - 요청/응답 형식 (Pydantic)
#   dependencies.py    - 서버가 공유하는 자원과 그걸 꺼내 쓰는 의존성 함수
#   routers/           - 엔드포인트 묶음 (/chat, /programs). 기능별로 파일을 나눠 main.py가 비대해지지 않게 함
#   chat_log.py        - TB_CHAT_LOG(SQLite) 저장/조회 (SFR-008)
#   program_service.py - /programs가 쓰는 OpenSearch 조회 로직
#   RAG 로직 자체(검색/생성/가드레일)는 여기서 다시 구현하지 않고 src/rag/pipeline.py를 그대로 부릅니다.
#   API 계층은 "HTTP 요청을 파이프라인 호출로 바꾸고, 결과를 응답 형식에 맞추는 일"만 합니다.
#
# [CORS 미들웨어를 안 넣은 이유] Streamlit(WBS 6.4)은 브라우저가 아니라 Streamlit 서버(파이썬)가
#   requests로 이 API를 부릅니다. CORS는 "브라우저의 자바스크립트"가 다른 출처를 호출할 때만 걸리는 제한이라
#   지금 구조에서는 필요 없습니다. 나중에 React 같은 브라우저 프론트엔드를 붙이면 그때 추가해야 합니다.

import logging
from contextlib import asynccontextmanager

import sqlite3

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import NotFoundError as OpenSearchNotFoundError
from opensearchpy.exceptions import TransportError as OpenSearchTransportError

from config import INDEX_NAME, get_client          # src/indexing/config.py (경로는 src/api/__init__.py에서 설정)
from embedder import embed_batch                    # src/indexing/embedder.py
from llm_client import LLM_MODEL, LLM_PROVIDER, get_llm_client  # src/rag/llm_client.py
from query_slots import fetch_candidate_values      # src/rag/query_slots.py

from .chat_log import init_db
from .dependencies import AppResources, get_resources
from .routers import chat, programs
from .schemas import HealthResponse

# 기존 src/ 스크립트들은 print로 진행 상황을 찍었지만, 서버는 여러 요청이 동시에 섞여 들어오므로
# 시각/레벨이 붙는 logging을 씁니다 (uvicorn 로그와 같은 콘솔에 함께 출력됨).
# 전체(root) 레벨을 WARNING으로 두고 우리 로거("api")만 INFO로 여는 이유: root를 INFO로 두면
# opensearch-py가 요청마다, huggingface가 모델 파일마다 한 줄씩 찍어서(실행해보니 시작 시에만 40줄 이상)
# 정작 봐야 할 우리 로그가 묻힙니다.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
# opensearch-py는 연결 실패 시 재시도할 때마다 WARNING + 전체 트레이스백을 찍습니다(OpenSearch를 꺼두고
# 실행해보니 요청 1번에 트레이스백 3개). 연결 실패는 아래 공통 예외 처리에서 이미 한 줄짜리 로그로 남기므로,
# 중복되는 트레이스백은 숨깁니다.
logging.getLogger("opensearch").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    yield 앞: 서버가 요청을 받기 "전에" 한 번 실행 / yield 뒤: 서버가 꺼질 때 한 번 실행.
    무거운 준비 작업을 여기서 끝내두면 첫 번째 사용자가 그 비용을 떠안지 않습니다.
    """
    os_client = get_client()

    # OpenSearch가 꺼져 있어도 서버는 뜨게 둡니다 - fetch_candidate_values()가 실패 시 하드코딩 스냅샷으로
    # 대체해주고, 실제 요청 시점에 503으로 안내합니다(/health로도 상태 확인 가능).
    # [한계] 서버 시작 뒤에 OpenSearch를 켜면 candidates는 스냅샷 값 그대로 남습니다. 지금은 region이 "서울"
    #   하나라 차이가 없지만, WBS 10(지역 확장) 이후에는 서버 재시작이 필요하다는 점을 기억해야 합니다.
    candidates = fetch_candidate_values(os_client)
    llm = get_llm_client()

    # 임베딩 모델 미리 로딩(warm-up): embedder.py는 첫 호출 때 ko-sroberta 모델을 메모리에 올립니다(수 초 소요).
    # 이걸 안 하면 서버 시작 후 첫 /chat 요청만 유독 느려서 "평균 응답시간 5초 이내"(분석모델 정의서 2.5)
    # 측정값을 왜곡합니다.
    logger.info("임베딩 모델 로딩 중 (첫 실행 시 수 초 소요)...")
    embed_batch(["warm-up"])

    # 대화 이력 테이블(TB_CHAT_LOG)이 없으면 생성. 첫 /chat 요청 때 만들면 동시에 들어온 두 요청이
    # 같이 CREATE를 시도할 수 있어서, 요청을 받기 전인 여기서 한 번만 합니다.
    # [WBS 8.1] 실패해도(디스크 권한, 파일 잠김 등) 서버는 띄웁니다. 대화 이력은 부가 기능이라 질의응답까지
    #   막을 이유가 없고, /chat은 저장 실패 시 chat_id=None으로, /chat/history는 503으로 알아서 대응합니다.
    try:
        init_db()
    except sqlite3.Error as e:
        logger.error("대화 이력 DB 초기화 실패 - 이력 저장/조회 없이 서버를 시작합니다: %s", e)

    app.state.resources = AppResources(os_client=os_client, llm_client=llm, candidates=candidates)
    logger.info("서버 준비 완료 - 후보값 %s / LLM %s", candidates, "사용 가능" if llm else "비활성(키 없음)")
    yield
    os_client.close()


app = FastAPI(
    title="LocalGovSupportRAG API",
    description="지자체 지원사업 RAG 챗봇 백엔드 (질의응답 / 지원사업 조회)",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(programs.router)


# ── 공통 예외 처리 ───────────────────────────────────────────────────
@app.exception_handler(OpenSearchConnectionError)
async def opensearch_unavailable_handler(request: Request, exc: OpenSearchConnectionError):
    """
    OpenSearch가 꺼져 있을 때 엔드포인트마다 try/except를 쓰지 않고 여기서 한 번에 503으로 바꿉니다.
    이게 없으면 사용자는 원인을 알 수 없는 500 Internal Server Error만 보게 됩니다.
    503(Service Unavailable)은 "요청은 정상인데 뒤에 있는 서버가 지금 준비가 안 됐다"는 뜻이라
    클라이언트(Streamlit)가 "잠시 후 다시 시도" 안내를 띄우기에 알맞습니다.
    """
    logger.error("OpenSearch 연결 실패: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "검색 서버(OpenSearch)에 연결할 수 없습니다. OpenSearch가 켜져 있는지 확인하세요."},
    )


# [WBS 8.1] 아래 두 처리는 위의 연결 실패와 같은 TransportError 계열입니다. FastAPI(Starlette)는 예외의 클래스
# 계층을 아래에서부터(가장 구체적인 것부터) 훑어서 처음 등록된 처리를 쓰므로, ConnectionError/NotFoundError는
# 각자의 처리로, 나머지 OpenSearch 오류는 TransportError 처리로 갑니다. 등록 순서와는 상관없습니다.
@app.exception_handler(OpenSearchNotFoundError)
async def index_missing_handler(request: Request, exc: OpenSearchNotFoundError):
    """
    검색 요청에서 NotFoundError가 나는 경우는 사실상 "인덱스가 아직 없음"뿐입니다 (문서 단건 조회 API를 쓰지 않고
    전부 search로 조회하므로). 새 PC에서 색인 단계를 건너뛰고 서버부터 띄웠을 때 나는 상황이라, 해결 순서를 알려줍니다.
    """
    logger.error("OpenSearch 인덱스 없음: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": f"검색 인덱스({INDEX_NAME})가 없습니다. src/indexing의 index_mapping.py -> embedder.py -> "
                           "bulk_indexer.py 순서로 색인한 뒤 다시 시도하세요."},
    )


@app.exception_handler(OpenSearchTransportError)
async def opensearch_error_handler(request: Request, exc: OpenSearchTransportError):
    """
    OpenSearch가 요청을 받긴 했지만 오류로 응답한 경우 (잘못된 쿼리 400, 서버 내부 오류 500 등).
    502(Bad Gateway) = "우리 서버는 정상인데, 뒤에서 부른 서버가 잘못된 응답을 줬다"는 뜻이라 원인 구분에 맞습니다.
    쿼리 내용 등 내부 정보는 로그에만 남기고 사용자에게는 일반 문구만 보여줍니다.
    """
    logger.error("OpenSearch 오류 응답 (status=%s): %s", exc.status_code, exc)
    return JSONResponse(status_code=502, content={"detail": "검색 서버가 오류를 반환했습니다. 잠시 후 다시 시도해 주세요."})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """
    위에서 처리하지 못한 모든 예외의 마지막 그물. 이게 없으면 FastAPI는 "Internal Server Error"라는 일반 텍스트를
    돌려주는데, 화면(ui/api_client.py)은 {"detail": ...} JSON을 기대하므로 형식을 맞춰 줍니다.
    logger.exception은 전체 트레이스백까지 남겨서, 사용자에게는 짧은 문구를 보여주되 원인 추적은 가능하게 합니다.
    """
    logger.exception("처리되지 않은 예외 (%s %s)", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."})


# ── 상태 점검 ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(res: AppResources = Depends(get_resources)):
    # ping()은 연결이 안 되면 예외 대신 False를 돌려주므로 위의 503 처리로 빠지지 않고 상태만 보고합니다.
    opensearch_ok = res.os_client.ping()
    index_docs = None
    if opensearch_ok:
        # 서버는 떠 있는데 인덱스가 없는 경우(색인 전)도 /health 자체는 실패시키지 않고 "degraded"로 보고합니다.
        # /health는 "무엇이 문제인지 알려주는 곳"이라, 여기서 503을 내면 화면이 원인을 구분할 수 없습니다.
        try:
            index_docs = res.os_client.count(index=INDEX_NAME)["count"]
        except OpenSearchNotFoundError:
            index_docs = None
    return HealthResponse(
        # LLM이 없어도 검색 결과는 돌려줄 수 있으므로(pipeline의 llm_unavailable), OpenSearch와 인덱스만 기준으로 판단
        status="ok" if opensearch_ok and index_docs is not None else "degraded",
        opensearch=opensearch_ok,
        index_docs=index_docs,
        llm_enabled=res.llm_client is not None,
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
    )

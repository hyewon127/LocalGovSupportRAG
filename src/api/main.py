# [WBS 6.1] FastAPI 앱 진입점 - 서버 시작/종료 처리, 공통 예외 처리, 라우터 등록
#
# 실행 방법 (프로젝트 루트에서, OpenSearch가 먼저 떠 있어야 함):
#   .venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
#   -> http://localhost:8000/docs 에서 API 문서(Swagger UI)를 보고 바로 호출해볼 수 있습니다.
#   포트 8000은 분석모델 정의서 1.2 "포트번호: 8000 (FastAPI)"를 따른 것입니다.
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

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from config import INDEX_NAME, get_client          # src/indexing/config.py (경로는 src/api/__init__.py에서 설정)
from embedder import embed_batch                    # src/indexing/embedder.py
from llm_client import LLM_MODEL, LLM_PROVIDER, get_llm_client  # src/rag/llm_client.py
from query_slots import fetch_candidate_values      # src/rag/query_slots.py

from .chat_log import init_db
from .dependencies import AppResources, get_resources
from .routers import chat
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
# 실행해보니 요청 1번에 트레이스백 3개). 연결 실패는 아래 503 처리와 routers/chat.py의 ping()에서 이미
# 한 줄짜리 로그로 남기므로, 중복되는 트레이스백은 숨깁니다.
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
    init_db()

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


# ── 상태 점검 ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(res: AppResources = Depends(get_resources)):
    # ping()은 연결이 안 되면 예외 대신 False를 돌려주므로 위의 503 처리로 빠지지 않고 상태만 보고합니다.
    opensearch_ok = res.os_client.ping()
    index_docs = res.os_client.count(index=INDEX_NAME)["count"] if opensearch_ok else None
    return HealthResponse(
        # LLM이 없어도 검색 결과는 돌려줄 수 있으므로(pipeline의 llm_unavailable), OpenSearch만 기준으로 판단
        status="ok" if opensearch_ok else "degraded",
        opensearch=opensearch_ok,
        index_docs=index_docs,
        llm_enabled=res.llm_client is not None,
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
    )

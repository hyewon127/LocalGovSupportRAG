# [WBS 6.2] /chat 엔드포인트 - 질문 1개를 받아 RAG 답변 + 근거 출처를 돌려주고, 대화 이력을 남깁니다.
#
# 이 파일이 하는 일은 딱 네 가지입니다 (RAG 로직 자체는 src/rag/pipeline.py가 이미 다 합니다):
#   1) 요청 검증 - 화면에서 보낸 필터 값이 실제 인덱스에 있는 값인지
#   2) 파이프라인 호출 + 소요 시간 측정
#   3) TB_CHAT_LOG에 저장 (SFR-008)
#   4) 결과를 schemas.ChatResponse 형식으로 맞춰서 반환
#
# [async def가 아니라 def로 쓴 이유 - FastAPI에서 자주 하는 실수]
#   pipeline.answer_question()은 OpenSearch 요청, 임베딩 계산, LLM API 호출을 전부 "동기(blocking)" 방식으로
#   합니다. 이걸 async def 안에서 부르면 그 요청이 끝날 때까지 서버의 이벤트 루프 전체가 멈춰서,
#   다른 사용자의 요청(/health 같은 가벼운 것까지)도 전부 기다리게 됩니다.
#   def로 선언하면 FastAPI가 알아서 별도 스레드 풀에서 실행해주므로 이 문제가 없습니다.

import logging
import sqlite3
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query

from pipeline import answer_question  # src/rag/pipeline.py (경로는 src/api/__init__.py에서 설정)

from ..chat_log import get_history, save_chat
from ..dependencies import AppResources, get_resources
from ..schemas import ChatHistoryResponse, ChatRequest, ChatResponse

logger = logging.getLogger("api.chat")
router = APIRouter(prefix="/chat", tags=["chat"])


def _validate_filters(req: ChatRequest, candidates: dict[str, list[str]]) -> None:
    """
    화면에서 보낸 region/categories가 인덱스에 실제로 있는 값인지 확인합니다.

    왜 검증하는가: 없는 값(예: region="강남구" - 현재 region_name은 "서울"뿐)을 그대로 term 필터에 넣으면
    OpenSearch는 에러 없이 0건을 돌려주고, 파이프라인은 그걸 "근거 없음(no_evidence)"으로 답합니다.
    사용자는 "관련 사업이 없구나"라고 오해하게 되는데, 실제 원인은 잘못된 필터 값입니다.
    그래서 조용히 틀린 답을 주는 대신 422로 "허용되는 값은 이것들"이라고 명확히 알려줍니다.
    (targets는 필터가 아니라 검색어 힌트라서 목록에 없는 값이어도 결과가 사라지지 않으므로 검증하지 않습니다.)
    """
    if req.region and req.region not in candidates["region"]:
        raise HTTPException(422, f"알 수 없는 지역입니다: {req.region} (가능한 값: {candidates['region']})")
    unknown = [c for c in req.categories if c not in candidates["category"]]
    if unknown:
        raise HTTPException(422, f"알 수 없는 분야입니다: {unknown} (가능한 값: {candidates['category']})")


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, res: AppResources = Depends(get_resources)):
    _validate_filters(req, res.candidates)

    # [WBS 8.1 변경] 예전에는 여기서 ping()으로 OpenSearch 연결을 먼저 확인했습니다. hybrid_search()가 연결 실패를
    #   빈 결과로 삼켜서 "근거 없음"으로 둔갑했기 때문인데, 이제 hybrid_search()가 예외를 그대로 올리므로
    #   main.py의 공통 예외 처리가 503(연결 실패)/503(인덱스 없음)/502(검색 오류)로 바꿔줍니다.
    #   ping을 빼서 요청마다 왕복 1번이 줄었고, "ping은 성공했는데 검색 직전에 끊긴" 경우도 같이 잡힙니다.

    session_id = req.session_id or uuid.uuid4().hex

    # time.time()이 아니라 perf_counter()를 쓰는 이유: 시스템 시계가 도중에 보정되어도 영향을 받지 않는,
    # "경과 시간 측정 전용" 시계라서 응답 시간 같은 구간 측정에 정확합니다.
    started = time.perf_counter()
    result = answer_question(
        req.question,
        os_client=res.os_client,
        llm_client=res.llm_client,
        candidates=res.candidates,
        region=req.region,
        categories=req.categories,
        target_keywords=req.targets,
    )
    response_time_ms = int((time.perf_counter() - started) * 1000)

    if result["status"] == "llm_error":
        # 원본 오류 메시지(키 오류, 네트워크 오류 등)는 서버 로그에만 남기고 응답에는 싣지 않습니다
        # (ChatResponse에 error 필드가 없어서 자동으로 빠짐 - schemas.py 상단 주석 2번).
        # 내부 설정 정보가 사용자 화면에 노출되지 않게 하기 위함입니다.
        logger.warning("LLM 호출 실패 (session=%s): %s", session_id, result.get("error"))

    # 로그 저장 실패(디스크 문제, DB 파일 잠김 등)가 사용자 응답을 막으면 안 됩니다 - 답변은 이미 만들어졌고,
    # 이력 저장은 부가 기능이기 때문입니다. 실패하면 로그만 남기고 chat_id=None으로 응답합니다.
    try:
        chat_id = save_chat(
            session_id=session_id,
            question=req.question,
            answer=result["answer"],
            status=result["status"],
            program_ids=[s["program_id"] for s in result["sources"]],
            response_time_ms=response_time_ms,
        )
    except sqlite3.Error as e:
        logger.error("대화 이력 저장 실패 (session=%s): %s", session_id, e)
        chat_id = None

    logger.info("chat session=%s status=%s sources=%d %dms",
                session_id, result["status"], len(result["sources"]), response_time_ms)

    return ChatResponse(
        chat_id=chat_id,
        session_id=session_id,
        status=result["status"],
        answer=result["answer"],
        sources=result["sources"],
        slots=result["slots"],
        response_time_ms=response_time_ms,
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
def chat_history(session_id: str, limit: int = Query(50, ge=1, le=200)):
    """
    SFR-008의 "재조회". 화면을 새로고침해도 같은 session_id로 이전 대화를 다시 그릴 수 있게 합니다.
    없는 session_id면 404가 아니라 빈 목록을 돌려줍니다 - "아직 대화가 없는 세션"도 정상 상태이기 때문입니다.
    """
    try:
        items = get_history(session_id, limit)
    except sqlite3.Error as e:
        # [WBS 8.1] DB 파일이 잠겼거나 서버 시작 시 초기화에 실패한 경우(main.py lifespan 참고).
        # 저장소 문제는 요청이 잘못된 게 아니라 서버 쪽 사정이라 503으로 알립니다.
        logger.error("대화 이력 조회 실패 (session=%s): %s", session_id, e)
        raise HTTPException(503, "대화 이력 저장소를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.") from e
    return ChatHistoryResponse(session_id=session_id, items=items)

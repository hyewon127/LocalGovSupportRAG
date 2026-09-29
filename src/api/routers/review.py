# [2026-09-29] /review - 대화 기록 검토 (사용자 요청: "채팅 기록들 보고 잘 됐는지 확인할 수 있는 화면")
#
# 화면: web/review.html (http://127.0.0.1:8000/review.html)
# 평가 스크립트(src/evaluation)가 "미리 만든 질문 60개"로 품질을 잰다면, 여기는 "실제로 들어온 질문"을 사람이 보고
# 판정하는 곳입니다. 평가셋에 없는 질문 유형(예: "청년 월세 지원" 같은 개인 복지 질문)을 발견하는 통로이기도 합니다.
#
# [인증이 없는 이유와 한계] 서버가 127.0.0.1에만 열려 있어서(scripts/start_chatbot.ps1) 이 PC 밖에서는 접근할 수 없습니다.
#   질문 원문이 그대로 보이는 화면이므로, 서버를 외부에 공개(0.0.0.0, 배포)할 때는 반드시 로그인을 먼저 붙여야 합니다.

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Query

from ..chat_log import chat_summary, clear_review, list_chats, set_review
from ..schemas import ReviewItem, ReviewListResponse, ReviewRequest, ReviewSummary

logger = logging.getLogger("api.review")
router = APIRouter(prefix="/review", tags=["review"])


def _storage_error(e: Exception) -> HTTPException:
    # chat.py의 대화 이력 조회와 같은 방침: 저장소 문제는 요청 탓이 아니라 서버 사정이라 503
    logger.error("대화 검토 저장소 오류: %s", e)
    return HTTPException(503, "대화 기록 저장소를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.")


@router.get("/summary", response_model=ReviewSummary)
def review_summary():
    try:
        return chat_summary()
    except sqlite3.Error as e:
        raise _storage_error(e) from e


@router.get("/chats", response_model=ReviewListResponse)
def review_chats(
    status: str | None = Query(None, description="ok / no_evidence / ungrounded / llm_unavailable / llm_error"),
    verdict: str | None = Query(None, pattern="^(good|bad|none)$", description="none = 아직 판정 안 한 대화"),
    q: str | None = Query(None, max_length=100, description="질문·답변 부분 검색"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        total, items = list_chats(status=status or None, verdict=verdict, query=(q or "").strip() or None,
                                  limit=limit, offset=offset)
    except sqlite3.Error as e:
        raise _storage_error(e) from e
    return ReviewListResponse(total=total, items=items)


@router.put("/chats/{chat_id}", response_model=ReviewItem)
def review_chat(chat_id: int, body: ReviewRequest):
    try:
        item = set_review(chat_id, body.verdict, body.note.strip())
    except sqlite3.Error as e:
        raise _storage_error(e) from e
    if item is None:
        raise HTTPException(404, f"대화 {chat_id}번이 없습니다.")
    return item


@router.delete("/chats/{chat_id}", status_code=204)
def unreview_chat(chat_id: int):
    """판정 취소 (대화 기록 자체는 지우지 않음)."""
    try:
        clear_review(chat_id)
    except sqlite3.Error as e:
        raise _storage_error(e) from e

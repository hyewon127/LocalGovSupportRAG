# [2026-09-29] /sync/status - 기업마당 자동 갱신(src/sync/sync_bizinfo.py)의 최근 실행 기록 조회
#
# 동기화는 API 서버와 별개 프로세스(작업 스케줄러가 매일 실행)라서, 둘 사이의 연결고리는 동기화가 남기는 기록 파일
# data/sync_state.json 하나뿐입니다. 이 API는 그 파일을 읽어 그대로 보여주기만 합니다 (서버가 동기화를 직접 돌리지 않는 이유:
# 임베딩·다운로드가 몇 분씩 걸려서 질문 응답과 같은 프로세스에서 돌리면 그동안 답변이 느려짐).
# 웹 화면 헤더의 "최근 갱신" 표시와, 데이터가 며칠째 안 바뀌었는지(스케줄러가 멈췄는지) 확인하는 용도.

import json
from pathlib import Path

from fastapi import APIRouter

from ..schemas import SyncStatus

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "sync_state.json"

router = APIRouter(prefix="/sync", tags=["system"])


@router.get("/status", response_model=SyncStatus)
def sync_status():
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        # 한 번도 안 돌린 상태(8월 배치 데이터만 있음)도 에러가 아니라 "기록 없음"으로 알려줌
        return SyncStatus(has_run=False)
    return SyncStatus(has_run=True, last_run=state.get("last_run"), last_success=state.get("last_success"),
                      recent_runs=state.get("runs", [])[:10])

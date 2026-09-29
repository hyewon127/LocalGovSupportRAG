# [WBS 6.2] 대화 이력 저장/조회 - 요구사항 SFR-008 "사용자 질의·응답 이력을 저장하고 재조회"
#
# 왜 SQLite인가 (테이블정의서 "테이블목록" 시트의 설계 메모를 그대로 따름):
#   지원사업 정보는 전부 OpenSearch 한 곳에 있지만, 대화 로그는 "검색 대상이 아닌 트랜잭션성 데이터"라
#   OpenSearch에 넣을 이유가 없습니다. SQLite는 파이썬에 기본 내장(sqlite3)이라 설치할 것도, 띄울 서버도
#   없고 파일 하나(data/chat_log.db)로 끝납니다 - 2주짜리 프로젝트의 로그 저장소로는 충분합니다.
#
# [테이블정의서와 다른 점 1개] STATUS 컬럼을 추가했습니다.
#   테이블정의서의 TB_CHAT_LOG에는 없는 컬럼이지만, pipeline.py가 돌려주는 status(ok/no_evidence/
#   ungrounded/llm_unavailable/llm_error)를 남기지 않으면 WBS 7.3(출처 인용률/hallucination 비율)을
#   로그로 계산할 때 "정상 답변"과 "근거 없음 안내"를 구분할 수 없습니다.
#   -> 테이블정의서에도 STATUS 컬럼을 추가해야 문서와 코드가 맞습니다 (config.py의 임베딩 차원 변경 때와 같은 상황).
#
# [연결을 매번 새로 여는 이유] FastAPI는 def 엔드포인트를 여러 스레드에서 동시에 실행합니다.
#   sqlite3 연결 객체는 기본적으로 "만든 스레드에서만" 쓸 수 있어서, 전역 연결 하나를 공유하면
#   "SQLite objects created in a thread can only be used in that same thread" 에러가 납니다.
#   함수 호출마다 열고 닫으면 이 문제가 없고, 로컬 파일 DB라 여는 비용도 1ms 수준이라 무시할 만합니다.

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/api/ -> src/ -> 프로젝트 루트

# 환경변수로 경로를 바꿀 수 있게 둔 이유: 테스트할 때 실제 로그 파일을 더럽히지 않고 임시 파일을 쓰기 위함.
DB_PATH = Path(os.getenv("CHAT_LOG_DB_PATH", _PROJECT_ROOT / "data" / "chat_log.db"))

# 컬럼명/타입은 테이블정의서 "챗봇 대화 이력" 시트 그대로 (STATUS만 추가).
# CREATED_AT의 CURRENT_TIMESTAMP는 SQLite 규칙상 UTC입니다 - 한국 시간으로 저장하지 않는 이유는
# 저장은 UTC로 통일하고 보여줄 때만 변환하는 게 표준이라서입니다 (서버가 해외 리전으로 옮겨가도 값이 안 바뀜).
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS TB_CHAT_LOG (
    CHAT_ID           INTEGER PRIMARY KEY AUTOINCREMENT,
    SESSION_ID        TEXT    NOT NULL,
    QUESTION          TEXT,
    ANSWER            TEXT,
    STATUS            TEXT,
    CITED_PROGRAM_IDS TEXT,
    RESPONSE_TIME_MS  INTEGER,
    CREATED_AT        TEXT    DEFAULT CURRENT_TIMESTAMP
)
"""
# 재조회(get_history)는 항상 "특정 세션의 로그를 순서대로"이므로 그 조건 그대로 인덱스를 겁니다.
# 로그가 쌓여도 전체 테이블을 훑지 않고 해당 세션만 바로 찾아갑니다.
_CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS IDX_CHAT_LOG_SESSION ON TB_CHAT_LOG (SESSION_ID, CHAT_ID)"

# [2026-09-29] 대화 검토 - 사람이 "잘 답했나"를 판정해서 남김 (웹 검토 화면 web/review.html).
# TB_CHAT_LOG에 컬럼을 추가하지 않고 테이블을 따로 둔 이유:
#   1) 대화 로그는 "그때 무슨 일이 있었나"의 원본 기록이라 바뀌면 안 되고, 판정은 사람이 몇 번이고 바꿀 수 있음
#   2) 이미 쌓인 로그 DB 파일에도 ALTER TABLE 없이 CREATE TABLE IF NOT EXISTS만으로 붙일 수 있음
# 대화 1건당 판정 1개(CHAT_ID가 기본키) - 다시 판정하면 덮어씀.
_CREATE_REVIEW_SQL = """
CREATE TABLE IF NOT EXISTS TB_CHAT_REVIEW (
    CHAT_ID     INTEGER PRIMARY KEY REFERENCES TB_CHAT_LOG(CHAT_ID),
    VERDICT     TEXT NOT NULL CHECK (VERDICT IN ('good', 'bad')),
    NOTE        TEXT,
    REVIEWED_AT TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 결과를 튜플 대신 컬럼명으로 꺼낼 수 있게 (row["QUESTION"])
    return conn


def init_db() -> None:
    """테이블이 없으면 만듭니다. IF NOT EXISTS라 서버를 여러 번 재시작해도 기존 로그는 그대로 남습니다."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # closing(): with 블록이 끝나면 연결을 닫음 / 안쪽 `with conn`: 블록이 끝나면 commit (에러 시 rollback).
    # sqlite3의 `with conn`은 "트랜잭션"만 관리하고 연결을 닫아주지는 않아서 둘을 같이 씁니다.
    with closing(_connect()) as conn, conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        conn.execute(_CREATE_REVIEW_SQL)


def save_chat(
    *,
    session_id: str,
    question: str,
    answer: str,
    status: str,
    program_ids: list[str],
    response_time_ms: int,
) -> int:
    """
    출력: 새로 저장된 행의 CHAT_ID

    program_ids는 응답에 실린 sources의 program_id입니다. status가 ok면 "답변이 실제로 인용한" 사업이고,
    llm_unavailable/llm_error면 "답변 대신 보여준 검색 결과"입니다 - 같은 컬럼에 담기지만 STATUS로 구분됩니다.
    """
    # 같은 사업의 청크가 2개까지 근거로 들어갈 수 있어서(hybrid_search.py MAX_CHUNKS_PER_PROGRAM),
    # 순서를 유지한 채 중복을 제거합니다. 테이블정의서 형식대로 콤마로 이어 붙여 TEXT 한 칸에 저장합니다.
    unique_ids = list(dict.fromkeys(program_ids))
    with closing(_connect()) as conn, conn:
        # 값은 반드시 ? 자리표시자로 넘깁니다. 질문 문자열을 f-string으로 SQL에 직접 끼워 넣으면
        # 사용자가 질문에 작은따옴표(')만 넣어도 SQL이 깨지고, 악의적인 입력이면 SQL 인젝션이 됩니다.
        cur = conn.execute(
            "INSERT INTO TB_CHAT_LOG (SESSION_ID, QUESTION, ANSWER, STATUS, CITED_PROGRAM_IDS, RESPONSE_TIME_MS) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, question, answer, status, ",".join(unique_ids), response_time_ms),
        )
        return cur.lastrowid


def get_history(session_id: str, limit: int = 50) -> list[dict]:
    """
    출력: 해당 세션의 로그를 오래된 것부터 최대 limit개 (schemas.ChatLogItem 형태의 dict 리스트)
    "최근 limit개"를 먼저 고른 뒤 다시 오래된 순으로 뒤집습니다 - 대화 화면은 위에서 아래로 시간순이어야
    하지만, 로그가 많을 때 잘라내야 하는 쪽은 가장 오래된 대화이기 때문입니다.
    """
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM TB_CHAT_LOG WHERE SESSION_ID = ? ORDER BY CHAT_ID DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

    return [_row_to_item(row) for row in reversed(rows)]


def _parse_utc(value: str | None) -> datetime | None:
    # SQLite의 "2026-09-22 01:23:45"(UTC, 시간대 표시 없음)에 UTC라고 명시해 줍니다. 이걸 안 하면
    # 받는 쪽에서 한국 시간으로 오해해서 9시간 어긋난 시각을 보여주게 됩니다.
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if value else None


def _row_to_item(row: sqlite3.Row) -> dict:
    return {
        "chat_id": row["CHAT_ID"],
        "session_id": row["SESSION_ID"],
        "question": row["QUESTION"] or "",
        "answer": row["ANSWER"] or "",
        "status": row["STATUS"] or "",
        "cited_program_ids": [p for p in (row["CITED_PROGRAM_IDS"] or "").split(",") if p],
        "response_time_ms": row["RESPONSE_TIME_MS"] or 0,
        "created_at": _parse_utc(row["CREATED_AT"]),
    }


# ── 대화 검토 (web/review.html, 2026-09-29) ──────────────────────────
_REVIEW_SELECT = """
SELECT L.*, R.VERDICT, R.NOTE, R.REVIEWED_AT
FROM TB_CHAT_LOG L LEFT JOIN TB_CHAT_REVIEW R ON R.CHAT_ID = L.CHAT_ID
"""


def _review_item(row: sqlite3.Row) -> dict:
    item = _row_to_item(row)
    item.update({"verdict": row["VERDICT"], "note": row["NOTE"] or "", "reviewed_at": _parse_utc(row["REVIEWED_AT"])})
    return item


def list_chats(*, status: str | None = None, verdict: str | None = None, query: str | None = None,
               limit: int = 50, offset: int = 0) -> tuple[int, list[dict]]:
    """
    출력: (조건에 맞는 전체 건수, 최신순 limit건)
    verdict: "good" / "bad" / "none"(아직 판정 안 함) / None(전체)
    query: 질문·답변에 들어간 글자 (LIKE 부분 일치 - 로그가 수만 건이 되기 전까지는 충분)
    조건 문장은 이어 붙이지만 값은 전부 ? 자리표시자로 넘김 (save_chat 주석의 SQL 인젝션 이유와 같음)
    """
    where, params = [], []
    if status:
        where.append("L.STATUS = ?")
        params.append(status)
    if verdict == "none":
        where.append("R.VERDICT IS NULL")
    elif verdict:
        where.append("R.VERDICT = ?")
        params.append(verdict)
    if query:
        where.append("(L.QUESTION LIKE ? OR L.ANSWER LIKE ?)")
        params += [f"%{query}%", f"%{query}%"]
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    with closing(_connect()) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM TB_CHAT_LOG L LEFT JOIN TB_CHAT_REVIEW R ON R.CHAT_ID = L.CHAT_ID{clause}", params
        ).fetchone()[0]
        rows = conn.execute(f"{_REVIEW_SELECT}{clause} ORDER BY L.CHAT_ID DESC LIMIT ? OFFSET ?",
                            [*params, limit, offset]).fetchall()
    return total, [_review_item(r) for r in rows]


def chat_summary() -> dict:
    """검토 화면 상단 요약: 상태별 건수, 판정 현황, 응답 시간, 날짜별 건수(한국 시간 기준 최근 14일)."""
    with closing(_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM TB_CHAT_LOG").fetchone()[0]
        by_status = dict(conn.execute("SELECT STATUS, COUNT(*) FROM TB_CHAT_LOG GROUP BY STATUS").fetchall())
        by_verdict = dict(conn.execute("SELECT VERDICT, COUNT(*) FROM TB_CHAT_REVIEW GROUP BY VERDICT").fetchall())
        times = [r[0] for r in conn.execute(
            "SELECT RESPONSE_TIME_MS FROM TB_CHAT_LOG WHERE RESPONSE_TIME_MS IS NOT NULL ORDER BY RESPONSE_TIME_MS")]
        # 저장은 UTC라서(_CREATE_TABLE_SQL 주석) 날짜로 묶을 때만 +9시간 - 한국 자정 기준으로 하루를 나눔
        by_day = conn.execute(
            "SELECT DATE(CREATED_AT, '+9 hours') AS D, COUNT(*) FROM TB_CHAT_LOG GROUP BY D ORDER BY D DESC LIMIT 14"
        ).fetchall()
    return {
        "total": total,
        "by_status": {k or "unknown": v for k, v in by_status.items()},
        "reviewed": sum(by_verdict.values()),
        "good": by_verdict.get("good", 0),
        "bad": by_verdict.get("bad", 0),
        "avg_response_ms": round(sum(times) / len(times)) if times else None,
        # p95: 느린 쪽 5% 경계. 평균은 몇 건의 아주 느린 응답(모델 첫 로딩 30초 등)에 끌려가서 체감과 다름
        "p95_response_ms": times[min(len(times) - 1, int(len(times) * 0.95))] if times else None,
        "by_day": [{"date": d, "count": c} for d, c in reversed(by_day)],
    }


def set_review(chat_id: int, verdict: str, note: str = "") -> dict | None:
    """판정 저장(이미 있으면 덮어씀). 없는 대화면 None."""
    with closing(_connect()) as conn, conn:
        if conn.execute("SELECT 1 FROM TB_CHAT_LOG WHERE CHAT_ID = ?", (chat_id,)).fetchone() is None:
            return None
        # 처음이면 추가, 이미 판정했으면 판정·메모·시각만 갱신 (SQLite 3.24+의 UPSERT)
        conn.execute(
            "INSERT INTO TB_CHAT_REVIEW (CHAT_ID, VERDICT, NOTE) VALUES (?, ?, ?) "
            "ON CONFLICT(CHAT_ID) DO UPDATE SET VERDICT = excluded.VERDICT, NOTE = excluded.NOTE, "
            "REVIEWED_AT = CURRENT_TIMESTAMP",
            (chat_id, verdict, note),
        )
        row = conn.execute(f"{_REVIEW_SELECT} WHERE L.CHAT_ID = ?", (chat_id,)).fetchone()
    return _review_item(row)


def clear_review(chat_id: int) -> bool:
    """판정 취소. 지운 게 있으면 True."""
    with closing(_connect()) as conn, conn:
        return conn.execute("DELETE FROM TB_CHAT_REVIEW WHERE CHAT_ID = ?", (chat_id,)).rowcount > 0

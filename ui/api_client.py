# [WBS 6.4] Streamlit 화면 -> FastAPI 백엔드 호출 모듈
#
# 왜 화면 코드(app.py)에서 requests를 직접 부르지 않고 이 파일로 모았는가:
#   1) 에러 처리를 한 곳에서: 백엔드 꺼짐 / OpenSearch 꺼짐(503) / 잘못된 입력(422)을 화면이 알아들을 수 있는
#      한국어 메시지 하나(ApiError)로 바꿔줍니다. 화면 코드 곳곳에 try/except와 상태코드 분기가 흩어지지 않습니다.
#   2) 주소/타임아웃 설정을 한 곳에서: API 주소가 바뀌어도 이 파일(또는 환경변수)만 고치면 됩니다.
#   3) 화면은 "무엇을 보여줄지"만, 이 파일은 "어떻게 가져올지"만 - 백엔드(src/api)를 routers와 service로
#      나눈 것과 같은 이유입니다.
#
# 화면이 src/rag를 직접 import해서 파이프라인을 부르지 않고 HTTP로 백엔드를 거치는 이유:
#   분석모델 정의서 1.1 시스템 구조도가 "클라이언트(웹/채팅 UI) -> FastAPI 서버 -> OpenSearch/LLM" 구조라서입니다.
#   화면이 파이프라인을 직접 부르면 대화 이력 저장(SFR-008), 입력 검증, 503 처리 같은 백엔드 로직을 우회하게 되고,
#   임베딩 모델도 Streamlit 프로세스에 한 벌 더 올라가 메모리를 두 배로 씁니다.

import os

import requests

# [127.0.0.1인 이유] Windows에서 "localhost"로 부르면 IPv6(::1)를 먼저 시도했다가 IPv4로 넘어가느라
# 요청마다 약 2초가 더 걸렸습니다 (실측: /health 14~30ms vs 2,058~2,069ms - src/api/main.py 상단 주석 참고).
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# /chat은 LLM 답변 생성까지 기다려야 해서 길게, 나머지 조회는 짧게.
# 타임아웃을 안 주면 백엔드가 멈췄을 때 화면도 영원히 "로딩 중"으로 멈춥니다 (requests 기본값은 무제한).
# 90초인 이유: 백엔드의 최악 대기 시간(src/rag/llm_client.py LLM_TIMEOUT_SEC 주석: 80초)보다 길어야
# 백엔드가 "LLM 오류 - 공고 목록만 안내"로 정상 응답할 기회를 화면이 먼저 끊어버리지 않습니다.
CHAT_TIMEOUT_SEC = 90
DEFAULT_TIMEOUT_SEC = 10


class ApiError(Exception):
    """화면에 그대로 보여줄 수 있는 한국어 메시지를 담은 예외."""


def _request(method: str, path: str, *, timeout: float = DEFAULT_TIMEOUT_SEC, **kwargs) -> dict:
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise ApiError(
            f"백엔드 서버({API_BASE_URL})에 연결할 수 없습니다. "
            "프로젝트 루트에서 `.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --port 8000`으로 먼저 실행하세요."
        ) from None
    except requests.Timeout:
        raise ApiError(f"백엔드 응답이 {timeout}초 안에 오지 않았습니다. 잠시 후 다시 시도해 주세요.") from None

    if response.ok:
        return response.json()

    # FastAPI의 에러 응답은 {"detail": ...} 형태입니다. HTTPException으로 직접 던진 건 detail이 문자열이고,
    # Pydantic 검증 실패(422)는 detail이 [{"loc":..., "msg":...}] 리스트라서 두 경우를 나눠 처리합니다.
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text
    if isinstance(detail, list):
        detail = " / ".join(item.get("msg", str(item)) for item in detail)
    raise ApiError(f"요청 실패 ({response.status_code}): {detail}")


def health() -> dict:
    return _request("GET", "/health", timeout=5)


def get_filters() -> dict:
    return _request("GET", "/programs/filters")


def count_programs() -> int:
    # size=1로 부르고 total만 씁니다 - 목록 전체를 받을 필요 없이 "사업 몇 건"만 알면 되므로.
    return _request("GET", "/programs", params={"size": 1})["total"]


def chat(question: str, *, session_id: str | None, region: str | None, categories: list[str], targets: list[str]) -> dict:
    payload = {"question": question, "session_id": session_id, "region": region,
               "categories": categories, "targets": targets}
    return _request("POST", "/chat", json=payload, timeout=CHAT_TIMEOUT_SEC)


def get_history(session_id: str) -> list[dict]:
    return _request("GET", f"/chat/history/{session_id}")["items"]


def get_program(program_id: str, max_chunks: int = 3) -> dict:
    return _request("GET", f"/programs/{program_id}", params={"max_chunks": max_chunks})

# [WBS 8.1] 예외 처리 테스트 - 빈 검색결과, OpenSearch/LLM/DB 오류 상황에서 사용자가 원인을 알 수 있는 응답이 나오는지
#
# 실행 (OpenSearch 실행 중 - 서버 시작 처리(lifespan)에 필요, 프로젝트 루트에서):
#   .venv\Scripts\python.exe tests\test_error_handling.py
#
# [어떻게 장애를 만드는가] 실제 OpenSearch를 끄면 다른 테스트와 동시에 돌릴 수 없고 느리므로, 장애 상황마다
#   "그 상황처럼 행동하는 가짜 클라이언트"(FakeOpenSearch)를 dependencies.get_resources 덮어쓰기로 끼워 넣습니다.
#   예: search()가 NotFoundError를 던지는 가짜 = "인덱스가 아직 없는 서버".
#   테스트 대상은 가짜가 아니라, 그 예외를 받은 우리 코드(pipeline -> API 예외 처리 -> 화면 메시지)가 어떻게 반응하는가입니다.

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["CHAT_LOG_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test_chat_log.db")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ui"))

import httpx  # noqa: E402
import openai  # noqa: E402
import requests  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from opensearchpy.exceptions import ConnectionError as OSConnectionError  # noqa: E402
from opensearchpy.exceptions import NotFoundError, RequestError  # noqa: E402

import api_client  # noqa: E402  (ui/api_client.py)
import src.api.main as api_main  # noqa: E402
from src.api.dependencies import AppResources, get_resources  # noqa: E402
from src.api.main import app  # noqa: E402
from src.api.routers import chat as chat_router  # noqa: E402

Q = "서울 소상공인 창업 자금 지원사업 알려줘"
results: list[tuple[str, bool]] = []


def check(name: str, condition, detail="") -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""))


class FakeOpenSearch:
    """ping()은 성공하지만 search()/count()가 지정한 동작(예외 또는 빈 결과)을 하는 가짜 클라이언트."""

    def __init__(self, search_error: Exception | None = None, count_error: Exception | None = None):
        self.search_error, self.count_error = search_error, count_error

    def ping(self):
        return True

    def search(self, index=None, body=None):
        if self.search_error:
            raise self.search_error
        return {"hits": {"hits": []}, "aggregations": {"program_count": {"value": 0}}}

    def count(self, index=None):
        if self.count_error:
            raise self.count_error
        return {"count": 0}


class TimeoutLLM:
    """답변 생성 호출에서 openai의 타임아웃 예외를 던지는 가짜 LLM."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.upstage.ai/v1/chat/completions"))


def use(real: AppResources, os_client=None, llm=None) -> None:
    app.dependency_overrides[get_resources] = lambda: AppResources(os_client or real.os_client, llm, real.candidates)


def test_api(c: TestClient, real: AppResources) -> None:
    print("\n── API 예외 처리 ──")

    # 1) 검색 도중 연결 끊김. 예전 코드는 hybrid_search가 연결 실패를 빈 결과로 삼켜서 "근거 없음"(200)이 나갔음
    use(real, os_client=FakeOpenSearch(search_error=OSConnectionError("N/A", "connection refused", None)))
    r = c.post("/chat", json={"question": Q})
    check("검색 중 연결 끊김 -> 503 (근거 없음으로 둔갑하지 않음)", r.status_code == 503, r.status_code)
    check("  메시지: OpenSearch 연결 안내", "OpenSearch" in r.json()["detail"], r.json())

    # 2) 인덱스 없음 (색인 전에 서버부터 띄운 경우)
    missing = NotFoundError(404, "index_not_found_exception", {"error": "no such index [idx_support_chunk]"})
    use(real, os_client=FakeOpenSearch(search_error=missing, count_error=missing))
    r = c.post("/chat", json={"question": Q})
    check("인덱스 없음 -> /chat 503 + 색인 순서 안내", r.status_code == 503 and "bulk_indexer" in r.json()["detail"], r.json())
    check("인덱스 없음 -> /programs 503", c.get("/programs").status_code == 503)
    h = c.get("/health")
    check("인덱스 없음 -> /health는 200 + degraded (원인 확인용이라 실패시키지 않음)",
          h.status_code == 200 and h.json()["status"] == "degraded" and h.json()["index_docs"] is None, h.json())

    # 3) OpenSearch가 요청은 받았지만 오류 응답 (잘못된 쿼리 등)
    use(real, os_client=FakeOpenSearch(search_error=RequestError(400, "search_phase_execution_exception", {})))
    r = c.post("/chat", json={"question": Q})
    check("검색 서버 오류 응답 -> 502", r.status_code == 502, r.status_code)
    check("  메시지에 내부 오류 원문 미노출", "search_phase_execution_exception" not in r.text)

    # 4) 예상하지 못한 예외 -> 형식을 맞춘 500 JSON (화면이 {"detail"}을 기대하므로)
    use(real, os_client=FakeOpenSearch(search_error=RuntimeError("unexpected secret detail")))
    r = c.post("/chat", json={"question": Q})
    check("예상 못 한 예외 -> 500 JSON detail", r.status_code == 500 and "detail" in r.json(), r.text[:100])
    check("  예외 원문 미노출", "unexpected secret detail" not in r.text)

    # 5) 빈 검색결과 (필터 조합에 맞는 문서가 0건 등) -> 정상 응답 + 근거 없음
    use(real, os_client=FakeOpenSearch())
    r = c.post("/chat", json={"question": Q})
    check("빈 검색결과 -> 200 no_evidence, 근거 0건",
          r.status_code == 200 and r.json()["status"] == "no_evidence" and r.json()["sources"] == [], r.json().get("status"))
    r = c.get("/programs", params={"q": "존재하지않는사업명"})
    check("빈 목록 조회 -> 200 total 0", r.status_code == 200 and r.json()["total"] == 0 and r.json()["items"] == [])

    # 6) LLM 타임아웃 -> 검색 결과는 살려서 llm_error로 안내
    use(real, llm=TimeoutLLM())
    r = c.post("/chat", json={"question": Q})
    body = r.json()
    check("LLM 타임아웃 -> 200 llm_error + 근거 목록 제공",
          r.status_code == 200 and body["status"] == "llm_error" and len(body["sources"]) > 0, body.get("status"))
    app.dependency_overrides.clear()

    # 7) 대화 이력 DB 오류 -> 조회는 503, 질의응답은 계속 동작
    original_get, original_save = chat_router.get_history, chat_router.save_chat

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    chat_router.get_history, chat_router.save_chat = broken, broken
    use(real, llm=None)
    check("이력 DB 오류 -> /chat/history 503", c.get("/chat/history/abc").status_code == 503)
    r = c.post("/chat", json={"question": Q})
    check("이력 DB 오류 -> /chat은 200 (chat_id=None)", r.status_code == 200 and r.json()["chat_id"] is None)
    chat_router.get_history, chat_router.save_chat = original_get, original_save
    app.dependency_overrides.clear()


def test_startup_without_db() -> None:
    print("\n── 서버 시작 시 DB 초기화 실패 ──")
    original = api_main.init_db

    def broken_init():
        raise sqlite3.OperationalError("attempt to write a readonly database")

    api_main.init_db = broken_init
    try:
        with TestClient(app) as c2:
            check("DB 초기화 실패해도 서버 시작 + /health 200", c2.get("/health").status_code == 200)
    except Exception as e:  # 서버가 아예 안 뜨면 여기로 옴
        check("DB 초기화 실패해도 서버 시작 + /health 200", False, repr(e))
    finally:
        api_main.init_db = original


def test_llm_client_settings() -> None:
    print("\n── LLM 클라이언트 타임아웃 설정 ──")
    import llm_client  # src/rag/llm_client.py (src.api import 시 경로가 잡힘)

    env_name = llm_client._SPEC["api_key_env"]
    saved = os.environ.get(env_name)
    os.environ[env_name] = "test-key-not-real"  # 네트워크 요청은 보내지 않음 - 클라이언트 객체 설정만 확인
    try:
        client = llm_client.get_llm_client()
        check(f"타임아웃 {llm_client.LLM_TIMEOUT_SEC}초 (openai 기본 600초 아님)", client.timeout == llm_client.LLM_TIMEOUT_SEC,
              client.timeout)
        check(f"재시도 {llm_client.LLM_MAX_RETRIES}회 (기본 2회 아님)", client.max_retries == llm_client.LLM_MAX_RETRIES)
        check("화면 타임아웃 > 백엔드 최악 대기(슬롯+답변 각각 timeout x (재시도+1))",
              api_client.CHAT_TIMEOUT_SEC > 2 * llm_client.LLM_TIMEOUT_SEC * (llm_client.LLM_MAX_RETRIES + 1),
              f"{api_client.CHAT_TIMEOUT_SEC}s")
    finally:
        if saved is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = saved


def test_ui_messages() -> None:
    print("\n── 화면(ui/api_client.py) 에러 메시지 변환 ──")
    original = requests.request

    def fake_response(status: int, body: bytes, content_type: str):
        r = requests.Response()
        r.status_code, r._content = status, body
        r.headers["Content-Type"] = content_type
        return r

    cases = {
        "500 텍스트 응답 -> 상태코드와 본문": (lambda *a, **k: fake_response(500, b"Internal Server Error", "text/plain"),
                                    lambda m: "500" in m and "Internal Server Error" in m),
        "502 JSON detail -> detail 문구": (lambda *a, **k: fake_response(502, '{"detail": "검색 서버가 오류를 반환했습니다."}'.encode(), "application/json"),
                                       lambda m: "502" in m and "검색 서버가 오류" in m),
        "422 검증 오류(리스트) -> msg 이어붙이기": (lambda *a, **k: fake_response(422, '{"detail": [{"msg": "질문이 비어 있습니다."}]}'.encode(), "application/json"),
                                       lambda m: "질문이 비어 있습니다" in m),
    }

    def raise_timeout(*a, **k):
        raise requests.Timeout()

    def raise_conn(*a, **k):
        raise requests.ConnectionError()

    cases["타임아웃 -> 잠시 후 재시도 안내"] = (raise_timeout, lambda m: "초 안에 오지 않았습니다" in m)
    cases["연결 불가 -> 백엔드 실행 방법 안내"] = (raise_conn, lambda m: "uvicorn src.api.main:app" in m)

    for name, (fake, ok) in cases.items():
        requests.request = fake
        try:
            api_client.health()
            check(name, False, "예외가 안 남")
        except api_client.ApiError as e:
            check(name, ok(str(e)), str(e))
        finally:
            requests.request = original


def main() -> int:
    with TestClient(app, raise_server_exceptions=False) as c:
        real = app.state.resources
        test_api(c, real)
    test_startup_without_db()
    test_llm_client_settings()
    test_ui_messages()

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

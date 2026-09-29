# [WBS 6.2~6.3] API 회귀 테스트 - /health, /chat, /chat/history, /programs 가 명세대로 동작하는지 확인
#
# 실행 방법 (OpenSearch만 켜져 있으면 됨 - 백엔드 서버는 따로 안 띄워도 됨, 프로젝트 루트에서):
#   .venv\Scripts\python.exe tests\test_api.py
#
# [왜 필요한가] WBS 6.2/6.3을 만들 때 같은 검사를 임시 스크립트로 돌렸는데, 레포에 남기지 않으면 나중에
#   pipeline.py나 schemas.py를 고쳤을 때 "예전에 되던 게 깨졌는지" 확인할 방법이 없습니다(회귀 테스트).
#   WBS 7에서 hybrid_search/pipeline을 건드리기 전에 이 파일로 기준선을 먼저 고정해 둡니다.
#
# [어떻게 테스트하는가]
#   - FastAPI TestClient: uvicorn을 띄우지 않고 같은 프로세스 안에서 앱에 요청을 보냅니다.
#     `with TestClient(app)`로 열어야 lifespan(서버 시작 처리: 임베딩 모델 로딩 등)까지 실제와 똑같이 실행됩니다.
#   - OpenSearch는 진짜 인덱스를 씁니다 (검색 결과가 실제 데이터 기준으로 맞는지 봐야 하므로).
#   - LLM은 가짜(FakeLLM)로 바꿉니다. 키가 없어도 ok/ungrounded/llm_error 경로를 확인할 수 있고, 실제 LLM은
#     답이 매번 달라서 테스트 결과가 흔들리기 때문입니다. dependencies.get_resources를 덮어써서 끼워 넣습니다.
#   - 대화 이력은 임시 DB 파일에 저장해서 실제 data/chat_log.db를 더럽히지 않습니다.
#   - 개수 기대값(사업 491건, 창업 41건 등)은 2026-09-22 인덱스(3,884청크) 기준입니다. 재색인하면 바뀔 수 있습니다.

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# chat_log.py가 import될 때 DB 경로를 정하므로, 반드시 앱을 import하기 "전에" 환경변수를 설정해야 합니다.
_TMP_DB = Path(tempfile.mkdtemp()) / "test_chat_log.db"
os.environ["CHAT_LOG_DB_PATH"] = str(_TMP_DB)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from opensearchpy import OpenSearch  # noqa: E402

from src.api.dependencies import AppResources, get_resources  # noqa: E402
from src.api.main import app  # noqa: E402
from src.api.routers import chat as chat_router  # noqa: E402

Q = "서울 소상공인 창업 자금 지원사업 알려줘"
results: list[tuple[str, bool]] = []


def check(name: str, condition, detail="") -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""))


class FakeLLM:
    """openai 클라이언트 중 generator.py가 쓰는 chat.completions.create()만 흉내 냅니다."""

    def __init__(self, content: str | None = None, exc: Exception | None = None):
        self.content, self.exc = content, exc
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if self.exc:
            raise self.exc
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def dead_opensearch() -> OpenSearch:
    # 아무것도 안 떠 있는 포트. max_retries=0: 재시도하면 실패 한 번에 수 초씩 걸려서 테스트가 느려짐
    return OpenSearch(hosts=[{"host": "localhost", "port": 9299}], use_ssl=False, timeout=1, max_retries=0)


def test_chat(c: TestClient, real: AppResources) -> None:
    print("\n── /chat, /chat/history (WBS 6.2) ──")

    def use(os_client=None, llm=None):
        app.dependency_overrides[get_resources] = lambda: AppResources(
            os_client or real.os_client, llm, real.candidates)

    # 실제 경로: LLM 키 없음 -> llm_unavailable (키가 있는 환경이면 LLM을 None으로 강제)
    use(llm=None)
    body = c.post("/chat", json={"question": Q}).json()
    check("실제 검색: llm_unavailable + 근거 5건", body["status"] == "llm_unavailable" and len(body["sources"]) == 5,
          body["status"])
    check("실제 검색: session_id 발급(32자) + chat_id 저장", len(body["session_id"]) == 32 and isinstance(body["chat_id"], int))
    check("실제 검색: 카드 필드(category/apply_period/amount/target)",
          all(k in body["sources"][0] for k in ("category", "apply_period", "amount", "target")))
    check("실제 검색: 내부 키(error/invalid_citations) 미노출", "error" not in body and "invalid_citations" not in body)
    check("실제 검색: 문장에서 슬롯 추출", body["slots"] == {"region": "서울", "categories": ["창업"],
                                                     "target_keywords": ["소상공인"]}, body["slots"])
    sid = body["session_id"]

    r = c.post("/chat", json={"question": "오늘 날씨 어때", "session_id": sid}).json()
    check("무관 질문: no_evidence + 근거 없음", r["status"] == "no_evidence" and r["sources"] == [], r["status"])

    items = c.get(f"/chat/history/{sid}").json()["items"]
    check("이력: 같은 세션 2건 시간순", [i["question"] for i in items] == [Q, "오늘 날씨 어때"])
    check("이력: created_at에 UTC 표기", items[0]["created_at"].endswith("Z"), items[0]["created_at"])
    check("이력: status 저장", [i["status"] for i in items] == ["llm_unavailable", "no_evidence"])
    ids = items[0]["cited_program_ids"]
    check("이력: 인용 사업 ID 중복 제거", len(ids) == len(set(ids)), ids)
    check("이력: 없는 세션은 404가 아니라 빈 목록", c.get("/chat/history/none").json()["items"] == [])

    check("422: 공백만 있는 질문", c.post("/chat", json={"question": "   "}).status_code == 422)
    check("422: 501자 질문", c.post("/chat", json={"question": "가" * 501}).status_code == 422)
    check("422: 인덱스에 없는 지역", c.post("/chat", json={"question": Q, "region": "강남구"}).status_code == 422)
    check("422: 인덱스에 없는 분야", c.post("/chat", json={"question": Q, "categories": ["우주"]}).status_code == 422)

    r = c.post("/chat", json={"question": Q, "categories": ["금융"], "targets": ["중소기업"]}).json()
    check("화면 필터: 분야는 덮어쓰기", r["slots"]["categories"] == ["금융"], r["slots"])
    check("화면 필터: 대상은 합치기", r["slots"]["target_keywords"] == ["소상공인", "중소기업"], r["slots"])
    check("화면 필터: 결과가 전부 금융", r["sources"] and all(s["category"] == "금융" for s in r["sources"]))

    use(llm=FakeLLM("첫 번째 사업을 추천합니다 [1]. 없는 번호 [9]."))
    r = c.post("/chat", json={"question": Q}).json()
    check("가짜 LLM ok: 인용한 근거만 남김", r["status"] == "ok" and [s["citation_index"] for s in r["sources"]] == [1],
          r["status"])
    check("가짜 LLM ok: 없는 인용 번호 제거", "[9]" not in r["answer"], r["answer"])

    use(llm=FakeLLM("인용 없이 지어낸 답변"))
    check("가짜 LLM: 인용 없으면 ungrounded", c.post("/chat", json={"question": Q}).json()["status"] == "ungrounded")

    use(llm=FakeLLM(exc=RuntimeError("secret-key-XYZ invalid")))
    r = c.post("/chat", json={"question": Q}).json()
    check("가짜 LLM 예외: llm_error + 검색 결과는 제공", r["status"] == "llm_error" and r["sources"], r["status"])
    check("가짜 LLM 예외: 오류 원문 미노출", "secret-key-XYZ" not in str(r))

    use(os_client=dead_opensearch())
    r = c.post("/chat", json={"question": Q})
    check("OpenSearch 꺼짐: /chat 503 (근거 없음으로 오안내하지 않음)", r.status_code == 503, r.status_code)
    h = c.get("/health").json()
    check("OpenSearch 꺼짐: /health degraded", h["status"] == "degraded" and h["opensearch"] is False)
    use(llm=None)

    original = chat_router.save_chat

    def broken_save(**kwargs):
        raise sqlite3.OperationalError("database is locked")

    chat_router.save_chat = broken_save
    r = c.post("/chat", json={"question": Q})
    check("이력 저장 실패해도 답변은 200 (chat_id=None)", r.status_code == 200 and r.json()["chat_id"] is None)
    chat_router.save_chat = original

    tricky = "창업'); DROP TABLE TB_CHAT_LOG; --"
    c.post("/chat", json={"question": tricky, "session_id": "sqli"})
    saved = c.get("/chat/history/sqli").json()["items"]
    check("SQL 문자열 질문: 원문 그대로 저장, 테이블 유지", saved and saved[0]["question"] == tricky)
    app.dependency_overrides.clear()


def test_programs(c: TestClient, real: AppResources) -> None:
    print("\n── /programs (WBS 6.3) ──")
    f = c.get("/programs/filters").json()
    check("filters: 지역/분야/대상 선택지", f["regions"] == ["서울"] and len(f["categories"]) == 8 and len(f["targets"]) == 7)

    r = c.get("/programs").json()
    ids = [i["program_id"] for i in r["items"]]
    check("목록: total은 사업 수(491, 청크 수 아님)", r["total"] == 491, r["total"])
    check("목록: 기본 20건, 같은 사업 중복 없음", len(ids) == 20 and len(set(ids)) == 20)
    dated = [i["apply_end"] for i in r["items"] if i["apply_end"]]
    check("목록: 마감일 내림차순", dated == sorted(dated, reverse=True))

    all_ids: set[str] = set()
    for page in range(1, 6):
        all_ids |= {i["program_id"] for i in c.get("/programs", params={"size": 100, "page": page}).json()["items"]}
    check("페이지: 5페이지 합치면 491개 고유 사업(겹침/누락 없음)", len(all_ids) == 491, len(all_ids))
    last = [i["apply_end"] for i in c.get("/programs", params={"size": 100, "page": 5}).json()["items"]]
    check("정렬: 마감일 없는 사업은 맨 뒤", last[-1] is None and all(e is None for e in last[last.index(None):]))

    r = c.get("/programs", params={"category": "창업", "size": 100}).json()
    check("필터: 창업 41건, 전부 창업", r["total"] == 41 and all(i["category"] == "창업" for i in r["items"]), r["total"])
    r = c.get("/programs", params={"q": "창업", "size": 100}).json()
    check("부분 검색: '창업' 31건(붙은 단어 포함)", r["total"] == 31 and all("창업" in i["program_name"] for i in r["items"]),
          r["total"])
    check("부분 검색: 대소문자 무시(ai -> AI 22건)", c.get("/programs", params={"q": "ai"}).json()["total"] == 22)
    check("부분 검색: '*'는 와일드카드가 아니라 글자 그대로", c.get("/programs", params={"q": "*"}).json()["total"] == 0)

    d = c.get("/programs/PBLN_000000000125563").json()
    check("상세: 청크 161개, 기본 앞 3개(0,1,2)", d["chunk_count"] == 161 and [x["seq"] for x in d["chunks"]] == [0, 1, 2])
    d = c.get("/programs/PBLN_000000000125563", params={"max_chunks": 200}).json()
    check("상세: 본문은 숫자 순서(_10이 _2보다 뒤)", [x["seq"] for x in d["chunks"]] == list(range(161)))
    check("상세: 없는 사업 404", c.get("/programs/PBLN_NOPE").status_code == 404)
    check("422: size=101 / page=0", c.get("/programs", params={"size": 101}).status_code == 422
          and c.get("/programs", params={"page": 0}).status_code == 422)

    app.dependency_overrides[get_resources] = lambda: AppResources(dead_opensearch(), None, real.candidates)
    check("OpenSearch 꺼짐: 목록/상세 503 (공통 예외 처리)",
          c.get("/programs").status_code == 503 and c.get("/programs/PBLN_000000000125563").status_code == 503)
    check("OpenSearch 꺼짐: filters는 서버 시작 시 값이라 200", c.get("/programs/filters").status_code == 200)
    app.dependency_overrides.clear()

    paths = list(c.get("/openapi.json").json()["paths"])
    check("OpenAPI: 엔드포인트 6개", paths == ["/chat", "/chat/history/{session_id}", "/programs/filters",
                                          "/programs", "/programs/{program_id}", "/health"], paths)


def main() -> int:
    with TestClient(app) as c:
        real = app.state.resources
        if not real.os_client.ping():
            print("OpenSearch가 꺼져 있습니다. 먼저 실행한 뒤 다시 시도하세요.")
            return 1
        test_chat(c, real)
        test_programs(c, real)

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} passed  (임시 DB: {_TMP_DB})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

# [2026-09-29] 대화 기록 검토 API(/review) 테스트 - OpenSearch·LLM 없이, 임시 SQLite 파일로 몇 초
#
# 실행: .venv\Scripts\python.exe tests\test_review.py
#
# 실제 data/chat_log.db를 건드리지 않도록 CHAT_LOG_DB_PATH를 임시 파일로 바꾼 뒤 import합니다(chat_log.py 주석).
# 앱 전체(src/api/main.py)는 시작할 때 OpenSearch·임베딩 모델을 올리므로, /review 라우터만 붙인 작은 앱으로 확인합니다.

import os
import sys
import tempfile
from pathlib import Path

TMP = tempfile.TemporaryDirectory()
os.environ["CHAT_LOG_DB_PATH"] = str(Path(TMP.name) / "review_test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import chat_log  # noqa: E402
from src.api.routers import review  # noqa: E402

results: list[bool] = []


def check(name: str, ok: bool, detail=None) -> None:
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok or detail is None else f"  ({detail})"))


def main() -> int:
    chat_log.init_db()
    ids = [
        chat_log.save_chat(session_id="s1", question="수출 지원사업 알려줘", answer="**A 사업** [1]", status="ok",
                           program_ids=["P1", "P1", "P2"], response_time_ms=5000),
        chat_log.save_chat(session_id="s1", question="청년 월세 지원", answer="찾지 못했습니다.", status="no_evidence",
                           program_ids=[], response_time_ms=40),
        chat_log.save_chat(session_id="s2", question="창업 지원", answer="…", status="llm_error",
                           program_ids=["P3"], response_time_ms=20000),
    ]
    app = FastAPI()
    app.include_router(review.router)
    c = TestClient(app)

    print("── 요약 ──")
    s = c.get("/review/summary").json()
    check("전체 3건, 상태별 집계", s["total"] == 3 and s["by_status"] == {"ok": 1, "no_evidence": 1, "llm_error": 1}, s)
    check("판정 전이라 reviewed 0", s["reviewed"] == 0 and s["good"] == 0 and s["bad"] == 0)
    check("평균 응답 시간", s["avg_response_ms"] == round((5000 + 40 + 20000) / 3), s["avg_response_ms"])

    print("\n── 목록·필터 ──")
    all_items = c.get("/review/chats").json()
    check("최신순", [i["chat_id"] for i in all_items["items"]] == ids[::-1])
    check("인용 공고 ID 중복 제거(P1 두 번 -> 한 번)", all_items["items"][-1]["cited_program_ids"] == ["P1", "P2"])
    check("상태 필터", [i["status"] for i in c.get("/review/chats?status=no_evidence").json()["items"]] == ["no_evidence"])
    check("검색(질문·답변 부분 일치)", c.get("/review/chats", params={"q": "월세"}).json()["total"] == 1)
    # 검색어를 SQL에 직접 끼워 넣었다면 이 입력이 쿼리를 깨뜨리거나 전체를 돌려줌 - 자리표시자라 0건이어야 정상
    check("SQL 인젝션 시도는 그냥 글자로 취급", c.get("/review/chats", params={"q": "' OR 1=1 --"}).json()["total"] == 0)
    page = c.get("/review/chats?limit=2&offset=2").json()
    check("페이지 나누기 (total은 전체 건수)", page["total"] == 3 and len(page["items"]) == 1)

    print("\n── 판정 ──")
    r = c.put(f"/review/chats/{ids[0]}", json={"verdict": "good", "note": "  공고 정확  "})
    check("판정 저장", r.status_code == 200 and r.json()["verdict"] == "good" and r.json()["note"] == "공고 정확", r.text)
    r = c.put(f"/review/chats/{ids[0]}", json={"verdict": "bad", "note": "기간 틀림"})
    check("다시 판정하면 덮어씀 (1건 유지)", r.json()["verdict"] == "bad" and c.get("/review/summary").json()["reviewed"] == 1)
    c.put(f"/review/chats/{ids[1]}", json={"verdict": "good"})
    check("판정 필터: 판정 전만", [i["chat_id"] for i in c.get("/review/chats?verdict=none").json()["items"]] == [ids[2]])
    check("판정 필터: 나쁨만", [i["chat_id"] for i in c.get("/review/chats?verdict=bad").json()["items"]] == [ids[0]])
    check("없는 판정 값은 422", c.put(f"/review/chats/{ids[0]}", json={"verdict": "maybe"}).status_code == 422)
    check("없는 대화는 404", c.put("/review/chats/9999", json={"verdict": "good"}).status_code == 404)
    check("판정 취소 204 + 집계 반영", c.delete(f"/review/chats/{ids[0]}").status_code == 204
          and c.get("/review/summary").json()["bad"] == 0)
    check("판정해도 원본 대화 기록은 그대로", chat_log.get_history("s1")[0]["answer"] == "**A 사업** [1]")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    code = main()
    TMP.cleanup()
    sys.exit(code)

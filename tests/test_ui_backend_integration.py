# [WBS 6.5] UI-백엔드 연동 테스트 - Streamlit 화면이 실제 FastAPI 백엔드와 붙어서 화면정의서대로 동작하는지 확인
#
# 실행 방법 (OpenSearch -> 백엔드 순서로 먼저 띄운 뒤, 프로젝트 루트에서):
#   .venv\Scripts\python.exe tests\test_ui_backend_integration.py
#   ※ 테스트 질문도 대화 이력(TB_CHAT_LOG)에 실제로 저장됩니다. 운영 로그와 섞고 싶지 않으면 백엔드를
#     CHAT_LOG_DB_PATH 환경변수로 다른 DB 파일을 지정해서 띄우세요 (src/api/chat_log.py 참고).
#
# [왜 이렇게 테스트하는가]
#   Streamlit 내장 테스트 도구(streamlit.testing.v1.AppTest)는 브라우저 없이 ui/app.py를 실행하고, 버튼 클릭/
#   채팅 입력을 코드로 흉내 낸 뒤 화면에 그려진 요소(제목, 캡션, 버튼 등)를 꺼내볼 수 있게 해줍니다.
#   백엔드는 가짜로 바꾸지 않고 실제 서버(127.0.0.1:8000)를 그대로 부르므로, "화면 -> HTTP -> FastAPI ->
#   OpenSearch"까지 한 번에 확인하는 진짜 연동 테스트입니다. 기대값도 하드코딩하지 않고 백엔드 API를 직접
#   불러서 얻은 값과 화면에 보인 값이 같은지 비교합니다 (데이터가 바뀌어도 테스트가 깨지지 않게).
#
#   이 프로젝트의 다른 검증 스크립트(src/indexing/verify_index.py 등)와 같은 방식으로, pytest 없이 바로 실행되고
#   항목마다 PASS/FAIL을 출력합니다.
#
# [범위 밖] 실제 브라우저에서의 모양(레이아웃/색)은 AppTest로 확인할 수 없어서, 화면은 직접 띄워서 눈으로 봐야 합니다.

import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "ui" / "app.py"
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

results: list[tuple[str, bool]] = []


def check(name: str, condition, detail="") -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""))


def new_app(query_params: dict | None = None) -> AppTest:
    # default_timeout: /chat 한 번(검색 + 임베딩)이 AppTest 기본값 3초를 넘을 수 있어서 넉넉하게
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    for key, value in (query_params or {}).items():
        at.query_params[key] = value
    return at.run()


def texts(elements) -> list[str]:
    return [e.value for e in elements]


def find_button(at: AppTest, *, label: str | None = None, key_prefix: str | None = None):
    for b in at.button:
        if (label and b.label == label) or (key_prefix and (b.key or "").startswith(key_prefix)):
            return b
    return None


def main() -> int:
    # ── 0. 사전 조건: 백엔드가 떠 있어야 함 ─────────────────────────
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
    except requests.RequestException:
        print(f"백엔드({API})가 꺼져 있습니다. 먼저 `python -m uvicorn src.api.main:app --port 8000`으로 실행하세요.")
        return 1
    filters = requests.get(f"{API}/programs/filters", timeout=10).json()
    program_total = requests.get(f"{API}/programs", params={"size": 1}, timeout=10).json()["total"]
    print(f"백엔드 상태: {health}\n")

    # ── 1. SCR-01 초기 화면 ─────────────────────────────────────────
    at = new_app()
    check("초기 화면: 예외 없음", not at.exception, texts(at.exception))
    check("헤더: 서비스명", texts(at.title) == ["🏛️ 지자체 지원사업 RAG 챗봇"])
    check("헤더: 공고 수가 백엔드 /programs total과 일치", any(f"{program_total:,}건" in c for c in texts(at.caption)))
    region_box, target_box = at.sidebar.selectbox
    check("필터: 지역 선택지 = 백엔드 값", region_box.options == ["전체"] + filters["regions"], region_box.options)
    check("필터: 업종 선택지 = 백엔드 값", target_box.options == ["전체"] + filters["targets"])
    check("필터: 관심분야 선택지 = 백엔드 값", at.sidebar.multiselect[0].options == filters["categories"])
    example = find_button(at, label="서울 소상공인 창업 자금 지원사업 알려줘")
    check("입력창: 추천 질문 버튼 표시", example is not None)
    check("입력창: 채팅 입력 활성화", at.chat_input and not at.chat_input[0].disabled)

    # ── 2. 추천 질문 -> 답변 + UI-02 카드 ─────────────────────────────
    example.click().run()
    check("질문 후: 예외 없음", not at.exception, texts(at.exception))
    check("대화창: 사용자/챗봇 말풍선", [m.name for m in at.chat_message] == ["user", "assistant"])
    expected_notice = "LLM API 키가 설정되지 않아" if not health["llm_enabled"] else None
    if expected_notice:
        check("상태 안내: LLM 키 없음 경고", any(expected_notice in w for w in texts(at.warning)), texts(at.warning))
    condition_caption = next((c for c in texts(at.caption) if c.startswith("🔎 검색 조건")), "")
    check("검색 조건 캡션: 질문에서 추출한 슬롯 표시", "지역 서울" in condition_caption and "분야 창업" in condition_caption,
          condition_caption)
    detail_buttons = [b for b in at.button if (b.key or "").startswith("detail_")]
    check("UI-02: 답변 카드(상세 보기 버튼) 1~5장", 1 <= len(detail_buttons) <= 5, len(detail_buttons))
    card_program_ids = [b.key.split("_", 2)[2] for b in detail_buttons]
    check("UI-02: 같은 사업은 카드 1장으로 합침", len(card_program_ids) == len(set(card_program_ids)), card_program_ids)
    session_id = at.session_state.session_id
    check("세션: 백엔드가 발급한 session_id를 URL에 저장", session_id and at.query_params.get("session") == [session_id],
          dict(at.query_params))

    # ── 3. SCR-02 상세 화면 ─────────────────────────────────────────
    first_program_id = card_program_ids[0]
    backend_detail = requests.get(f"{API}/programs/{first_program_id}", timeout=10).json()
    detail_buttons[0].click().run()
    check("상세: 예외 없음", not at.exception, texts(at.exception))
    headers = texts(at.header)
    check("상세: 사업명 = 백엔드 /programs/{id}",
          headers and headers[0].replace("\\", "") == backend_detail["program_name"], headers)
    check("상세: 핵심 정보 5칸(지자체/분야/접수기간/지원대상/지원금액)",
          all(label in texts(at.caption) for label in ["지자체", "분야", "접수기간", "지원대상", "지원금액"]))
    check("상세: 돌아가기 버튼", find_button(at, label="◀ 대화로 돌아가기") is not None)
    shown = min(3, backend_detail["chunk_count"])
    body_caption = next((c for c in texts(at.caption) if c.startswith("전체 ")), "")
    check("상세: 본문 앞 3구간 표시", f"앞 {shown}개" in body_caption, body_caption)
    more = find_button(at, label="본문 더 보기 (+5)")
    if backend_detail["chunk_count"] > 3:
        more.click().run()
        expected = min(8, backend_detail["chunk_count"])
        body_caption = next((c for c in texts(at.caption) if c.startswith("전체 ")), "")
        check("상세: 본문 더 보기 -> +5구간", f"앞 {expected}개" in body_caption, body_caption)
    else:
        check("상세: 구간이 3개 이하면 더 보기 버튼 없음", more is None)

    # ── 4. 돌아가기 -> 대화 유지 ────────────────────────────────────
    find_button(at, label="◀ 대화로 돌아가기").click().run()
    check("돌아가기: 대화 내역 유지", [m.name for m in at.chat_message] == ["user", "assistant"])

    # ── 5. 좌측 필터 적용 (관심분야=금융, 문장의 "창업"보다 우선) ───
    at.sidebar.multiselect[0].select("금융").run()
    at.chat_input[0].set_value("서울 소상공인 창업 자금 지원사업 알려줘").run()
    check("필터 질문: 예외 없음", not at.exception, texts(at.exception))
    captions = texts(at.caption)
    last_condition = [c for c in captions if c.startswith("🔎 검색 조건")][-1]
    check("필터 질문: 분야가 화면 선택값(금융)으로 덮어써짐", "분야 금융" in last_condition and "창업" not in last_condition.split("분야 ")[1].split(" ·")[0],
          last_condition)
    last_cards = [b.key.split("_", 2)[2] for b in at.button if (b.key or "").startswith("detail_3_")]
    categories = {requests.get(f"{API}/programs/{pid}", params={"max_chunks": 1}, timeout=10).json()["category"]
                  for pid in last_cards}
    check("필터 질문: 카드의 사업이 전부 금융 분야", last_cards and categories == {"금융"}, categories)
    at.sidebar.multiselect[0].unselect("금융").run()

    # ── 6. 새로고침(새 세션) -> URL의 session으로 대화 이력 복원 (SFR-008) ─
    history = requests.get(f"{API}/chat/history/{session_id}", timeout=10).json()["items"]
    reloaded = new_app({"session": session_id})
    check("새로고침: 예외 없음", not reloaded.exception, texts(reloaded.exception))
    check("새로고침: 백엔드 이력 수만큼 대화 복원", len(reloaded.chat_message) == 2 * len(history),
          f"화면 {len(reloaded.chat_message)} / 이력 {len(history)}턴")
    check("새로고침: 이전 대화 표시", any("이전 대화 기록" in c for c in texts(reloaded.caption)))

    # ── 7. 새 대화 시작 ─────────────────────────────────────────────
    find_button(reloaded, label="🆕 새 대화 시작").click().run()
    check("새 대화: 대화창 비움", len(reloaded.chat_message) == 0)
    check("새 대화: URL의 session 제거", "session" not in dict(reloaded.query_params), dict(reloaded.query_params))
    check("새 대화: 추천 질문 다시 표시", find_button(reloaded, label="서울 소상공인 창업 자금 지원사업 알려줘") is not None)

    # ── 8. 근거 없는 질문 -> no_evidence 안내, 카드 없음 ───────────────
    reloaded.chat_input[0].set_value("오늘 날씨 어때").run()
    check("무관 질문: 근거 없음 안내", any("관련 공고를 찾지 못했어요" in i for i in texts(reloaded.info)), texts(reloaded.info))
    check("무관 질문: 카드 없음", not [b for b in reloaded.button if (b.key or "").startswith("detail_")])

    # ── 9. 백엔드 꺼짐 -> 실행 안내 후 멈춤 ─────────────────────────
    # AppTest는 같은 파이썬 프로세스에서 app.py를 다시 실행하므로, 이미 import된 api_client 모듈의 주소를
    # 없는 포트로 바꿔서 "백엔드 꺼짐"을 흉내 냅니다. 캐시된 /health 결과가 남아 있으면 요청 자체를 안 보내므로
    # 캐시도 비웁니다.
    api_client = sys.modules["api_client"]
    original_url = api_client.API_BASE_URL
    api_client.API_BASE_URL = "http://127.0.0.1:8999"
    st.cache_data.clear()
    down = new_app()
    check("백엔드 꺼짐: 실행 방법 안내", any("uvicorn src.api.main:app" in e for e in texts(down.error)), texts(down.error))
    check("백엔드 꺼짐: 입력창을 그리지 않고 멈춤", len(down.chat_input) == 0)
    api_client.API_BASE_URL = original_url
    st.cache_data.clear()

    # ── 10. 실제 실행 명령으로 Streamlit 서버 기동 ────────────────────
    # AppTest는 스크립트를 직접 실행할 뿐이라, 실제 "streamlit run" 명령이 뜨는지도 따로 확인합니다.
    port = 8599  # 사용자가 띄워둔 8501과 겹치지 않게
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless", "true", "--server.port", str(port)],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ok = False
        for _ in range(60):
            try:
                ok = requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).text == "ok"
                if ok:
                    break
            except requests.RequestException:
                time.sleep(0.5)
        check("streamlit run: 서버 기동(/_stcore/health)", ok)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

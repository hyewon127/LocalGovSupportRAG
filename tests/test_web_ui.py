# [WBS 9] 챗봇 웹 화면(web/) 테스트
#
# 실행 (백엔드 실행 중, 프로젝트 루트에서): .venv\Scripts\python.exe tests\test_web_ui.py
#   tests/run_all.py가 백엔드를 띄운 뒤 같이 돌립니다.
#
# 세 단계로 확인합니다:
#   1. 화면 로직(web/chat-format.js) - Node로 순수 함수 테스트 (XSS 방어, 답변 형식 변환). Node가 없으면 건너뜀
#   2. 서빙 - 백엔드가 "/"에서 화면 파일을 주는지, 그리고 화면 서빙이 API 경로(/health, /chat)를 가로채지 않는지
#      (src/api/main.py에서 정적 파일을 "맨 끝에" 붙인 이유를 검증)
#   3. 실제 브라우저 - headless Edge로 페이지를 열어 JS가 실행된 뒤의 DOM에 인사말·추천 질문이 그려졌는지.
#      Streamlit 때는 내용이 웹소켓으로 늦게 와서 headless 캡처가 빈 화면이었는데, 이 화면은 일반 HTTP라 확인 가능.

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"  # localhost 대신 127.0.0.1 (Windows IPv6 지연 - src/api/main.py 주석)
EDGE_PATHS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
              Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]

results: list[bool] = []


def check(name: str, ok: bool, detail=None) -> None:
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok or detail is None else f"  ({detail})"))


def test_format_logic() -> None:
    print("── 1. 화면 로직 (Node) ──")
    node = shutil.which("node")
    if not node:
        print("[SKIP] node가 없어 건너뜀")
        return
    proc = subprocess.run([node, "tests/web_format.test.mjs"], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    print("\n".join(line for line in proc.stdout.splitlines() if line.startswith("[")))
    check("web/chat-format.js 단위 테스트 전부 통과", proc.returncode == 0, proc.stdout[-200:] + proc.stderr[-200:])


def test_serving() -> None:
    print("\n── 2. 서빙 ──")
    page = requests.get(f"{API}/", timeout=10)
    check("GET / -> index.html", page.ok and "text/html" in page.headers.get("content-type", "") and 'id="messages"' in page.text,
          page.status_code)
    for asset, kind in (("app.js", "javascript"), ("chat-format.js", "javascript"), ("style.css", "css"),
                        ("review.html", "html"), ("review.js", "javascript"), ("review.css", "css")):
        r = requests.get(f"{API}/{asset}", timeout=10)
        check(f"GET /{asset} -> {kind}", r.ok and kind in r.headers.get("content-type", ""), r.headers.get("content-type"))
    health = requests.get(f"{API}/health", timeout=10)
    check("화면 서빙이 API를 가로채지 않음: /health는 여전히 JSON", health.ok and health.json().get("status") in ("ok", "degraded"))
    bad = requests.post(f"{API}/chat", json={"question": ""}, timeout=10)
    check("화면 서빙이 API를 가로채지 않음: 빈 질문 /chat -> 422 JSON", bad.status_code == 422 and "detail" in bad.json())
    check("없는 파일은 404 (index.html로 덮지 않음)", requests.get(f"{API}/no-such-file.js", timeout=10).status_code == 404)


def test_browser() -> None:
    print("\n── 3. 실제 브라우저 (headless Edge) ──")
    edge = next((p for p in EDGE_PATHS if p.exists()), None)
    if edge is None:
        print("[SKIP] Edge가 없어 건너뜀")
        return
    with tempfile.TemporaryDirectory() as profile:
        # --virtual-time-budget: 페이지의 JS(fetch로 /health 등 호출)가 끝날 때까지 기다린 뒤 DOM을 출력
        proc = subprocess.run([str(edge), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                               "--virtual-time-budget=8000", "--dump-dom", f"{API}/"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    dom = proc.stdout
    check("JS 실행 후 인사말 말풍선이 그려짐", "지원사업 도우미예요" in dom, dom[:200])
    check("추천 질문 칩 4개", dom.count('class="chip"') >= 4, dom.count('class="chip"'))
    check("헤더 상태가 '연결 중'에서 바뀜 (/health 호출 성공)", 'data-state="loading"' not in dom)
    check("헤더에 공고 수 표시 (/programs 호출 성공)", "기업마당 공고" in dom and "건 · 서울" in dom)


def main() -> int:
    try:
        requests.get(f"{API}/health", timeout=5)
    except requests.RequestException:
        print(f"백엔드({API})가 꺼져 있습니다. 먼저 실행하세요.")
        return 1
    test_format_logic()
    test_serving()
    test_browser()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

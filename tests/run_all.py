# [WBS 8.2] 전체 통합 테스트 실행기 - 명령 하나로 모든 테스트를 순서대로 돌리고 요약표를 출력
#
# 실행 (OpenSearch만 먼저 켜두면 됨 - 백엔드 서버는 이 스크립트가 알아서 띄우고 끔, 프로젝트 루트에서):
#   .venv\Scripts\python.exe tests\run_all.py            # 전체 (약 4~5분)
#   .venv\Scripts\python.exe tests\run_all.py --quick    # 오래 걸리는 2개(데이터 파이프라인, 검색 결정성) 제외
#
# [왜 필요한가]
#   테스트가 8개 파일로 나뉘어 있고, 각자 필요한 환경(없음 / OpenSearch / 백엔드 서버)이 달라서 손으로 돌리면
#   순서나 서버 띄우기를 빠뜨리기 쉽습니다. 특히 UI 연동 테스트는 백엔드가 떠 있어야 하는데, 이걸 잊고 돌리면
#   "코드가 깨졌다"와 "서버를 안 띄웠다"가 구분되지 않습니다. 이 스크립트가 환경 준비까지 맡아서
#   "전부 통과 = 수집 산출물부터 화면까지 전 구간이 맞물려 동작한다"는 한 줄 결론을 낼 수 있게 합니다.
#
# [테스트 구성 - 아래로 갈수록 범위가 넓어짐]
#   1) 지표 계산 단위 테스트      : 순수 계산 (환경 불필요)
#   2) 데이터 파이프라인 정합성   : 수집 -> 추출 -> 청킹 -> 임베딩 -> 색인 산출물 대조
#   3) API 회귀 / 4) 예외 처리    : FastAPI 앱을 같은 프로세스에서 호출 (TestClient)
#   5) 검색 결정성 / 6) 품질 게이트 / 7) 답변 평가 경로 : 검색·평가 파이프라인
#   8) UI-백엔드 연동            : 실제 uvicorn 서버를 띄우고 Streamlit 화면을 HTTP로 연결

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
API_URL = "http://127.0.0.1:8000"  # localhost가 아니라 127.0.0.1 - Windows에서 요청마다 2초 지연 (src/api/main.py 주석)

# (파일, 필요한 환경, 오래 걸림 여부)
SUITES = [
    ("test_evaluation_metrics.py", "none", False),
    ("test_data_pipeline.py", "opensearch", True),
    ("test_api.py", "opensearch", False),
    ("test_error_handling.py", "opensearch", False),
    ("test_search_determinism.py", "opensearch", True),
    ("test_retrieval_quality.py", "opensearch", False),
    ("test_answer_eval.py", "opensearch", False),
    ("test_ui_backend_integration.py", "backend", False),
]


def opensearch_up() -> bool:
    try:
        return requests.get("http://localhost:9200", timeout=3).ok
    except requests.RequestException:
        return False


def backend_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def start_backend(log_dir: Path) -> subprocess.Popen | None:
    """
    uvicorn을 자식 프로세스로 띄우고 /health가 응답할 때까지 기다립니다(임베딩 모델 로딩 때문에 20초 안팎).
    대화 이력은 임시 DB에 쓰게 해서, 테스트 질문이 실제 data/chat_log.db에 섞이지 않게 합니다.
    """
    env = {**os.environ, "CHAT_LOG_DB_PATH": str(log_dir / "chat_log.db"), "PYTHONIOENCODING": "utf-8"}
    log = (log_dir / "backend.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", "8000"],
                            cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if proc.poll() is not None:  # 뜨다가 죽음 (import 에러 등)
            return None
        if backend_up():
            return proc
        time.sleep(1)
    proc.terminate()
    return None


def run_suite(filename: str, log_dir: Path) -> dict:
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, str(TESTS_DIR / filename)], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    output = proc.stdout + proc.stderr
    (log_dir / f"{filename}.log").write_text(output, encoding="utf-8")
    # 각 테스트는 마지막에 "N/M passed"를 출력하고, 한 줄짜리 테스트는 [PASS]/[FAIL]만 출력합니다.
    summary = re.findall(r"(\d+)/(\d+) passed", output)
    if summary:
        passed, total = map(int, summary[-1])
    else:
        passed, total = output.count("[PASS]"), output.count("[PASS]") + output.count("[FAIL]")
    return {"file": filename, "ok": proc.returncode == 0 and total > 0 and passed == total, "passed": passed,
            "total": total, "seconds": time.perf_counter() - started,
            "failures": [line for line in output.splitlines() if line.startswith("[FAIL]")][:5],
            "tail": "\n".join(output.strip().splitlines()[-8:])}


def main() -> int:
    parser = argparse.ArgumentParser(description="전체 통합 테스트 실행기 (WBS 8.2)")
    parser.add_argument("--quick", action="store_true", help="오래 걸리는 테스트(데이터 파이프라인, 검색 결정성) 제외")
    args = parser.parse_args()

    log_dir = Path(tempfile.mkdtemp(prefix="lgsrag_tests_"))
    has_opensearch = opensearch_up()
    print(f"OpenSearch: {'실행 중' if has_opensearch else '꺼져 있음 -> OpenSearch가 필요한 테스트는 건너뜀'}")
    print(f"로그 폴더: {log_dir}\n")

    rows, backend_proc, started_backend = [], None, False
    try:
        for filename, needs, slow in SUITES:
            if args.quick and slow:
                rows.append({"file": filename, "skipped": "--quick"})
                continue
            if needs in ("opensearch", "backend") and not has_opensearch:
                rows.append({"file": filename, "skipped": "OpenSearch 꺼짐"})
                continue
            if needs == "backend" and not backend_up():
                print("백엔드 서버 시작 중 (임베딩 모델 로딩 약 20초)...")
                backend_proc = start_backend(log_dir)
                if backend_proc is None:
                    rows.append({"file": filename, "skipped": f"백엔드 시작 실패 - {log_dir / 'backend.log'} 확인"})
                    continue
                started_backend = True
            print(f"실행: {filename} ...", flush=True)
            result = run_suite(filename, log_dir)
            rows.append(result)
            print(f"  -> {'통과' if result['ok'] else '실패'} {result['passed']}/{result['total']} ({result['seconds']:.0f}초)")
    finally:
        # 이 스크립트가 띄운 서버만 끕니다 (사용자가 원래 띄워둔 서버는 건드리지 않음).
        if started_backend and backend_proc is not None:
            backend_proc.terminate()
            backend_proc.wait(timeout=15)

    print("\n| 테스트 | 결과 | 통과/전체 | 시간 |\n|---|---|---|---|")
    for r in rows:
        if "skipped" in r:
            print(f"| {r['file']} | 건너뜀 ({r['skipped']}) | - | - |")
        else:
            print(f"| {r['file']} | {'통과' if r['ok'] else '**실패**'} | {r['passed']}/{r['total']} | {r['seconds']:.0f}초 |")

    ran = [r for r in rows if "skipped" not in r]
    failed = [r for r in ran if not r["ok"]]
    skipped = [r for r in rows if "skipped" in r]
    print(f"\n합계: 테스트 {len(ran)}개 실행 / 검사 항목 {sum(r['passed'] for r in ran)}/{sum(r['total'] for r in ran)} 통과 / "
          f"실패 {len(failed)}개 / 건너뜀 {len(skipped)}개")
    for r in failed:
        print(f"\n[실패] {r['file']}\n" + ("\n".join(r["failures"]) or r["tail"]))

    # 하나라도 실패하거나 건너뛰었으면 0이 아닌 코드로 끝냅니다 - "전체" 통합 테스트라서, 건너뛴 게 있으면
    # 전 구간을 확인했다고 말할 수 없기 때문입니다 (--quick은 사용자가 일부러 뺀 것이라 예외).
    incomplete = [r for r in skipped if r["skipped"] != "--quick"]
    return 0 if not failed and not incomplete else 1


if __name__ == "__main__":
    sys.exit(main())

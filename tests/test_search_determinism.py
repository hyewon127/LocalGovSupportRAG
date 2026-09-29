# [WBS 7.2에서 발견한 버그의 재현 테스트] 같은 질문이면 실행할 때마다 같은 top-5가 나와야 한다
#
# 실행 (OpenSearch 실행 중, 프로젝트 루트에서, 약 1분 소요):
#   .venv\Scripts\python.exe tests\test_search_determinism.py
#
# [배경] hybrid_search()가 점수만으로 정렬하던 시절, 중복 등록된 공고(원공고/변경공고)처럼 점수가 완전히 같은
#   청크끼리는 set 순회 순서대로 남았습니다. 파이썬은 프로세스마다 문자열 해시 시드를 바꾸므로(PYTHONHASHSEED)
#   set 순서도 실행마다 달라져서, 운영 설정에서도 Q07·Q17의 top-5가 실행마다 바뀌었습니다
#   (수정 전 코드로 확인: 질의x설정 66개 중 9개 불일치 -> 수정 후 0개).
#
# [테스트 방법] 한 프로세스 안에서 두 번 부르면 해시 시드가 같아서 버그가 안 보입니다. 그래서 해시 시드를
#   다르게 준 자식 프로세스 2개에서 평가셋 전체(33개)를 검색하고 결과를 비교합니다.

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CHILD = r'''
import contextlib, io, json, sys
root = sys.argv[1]
sys.path.insert(0, root + r"\src\rag")
sys.path.insert(0, root + r"\src\indexing")
with contextlib.redirect_stdout(io.StringIO()):  # 검색 모듈의 진행 로그가 결과 JSON과 섞이지 않게
    from config import get_client
    from hybrid_search import hybrid_search
    client = get_client()
    queries = json.load(open(root + r"\data\eval\eval_queries.json", encoding="utf-8"))["queries"]
    result = {q["id"]: [c["chunk_id"] for c in hybrid_search(q["question"], client=client)] for q in queries}
print(json.dumps(result))
'''


def run_with_seed(seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(PROJECT_ROOT)], capture_output=True, text=True,
                          encoding="utf-8", env=env, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    first, second = run_with_seed("1"), run_with_seed("2")
    different = [qid for qid in first if first[qid] != second[qid]]
    ok = not different and len(first) > 0
    print(f"[{'PASS' if ok else 'FAIL'}] 해시 시드가 달라도 top-5 동일  (질의 {len(first)}개 중 불일치 {len(different)}개 {different})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

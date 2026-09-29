# [2026-09-29] 기업마당 자동 갱신(src/sync/sync_bizinfo.py) 단위 테스트 - 인터넷·OpenSearch 없이 몇 초
#
# 실행: .venv\Scripts\python.exe tests\test_sync.py
#
# 동기화는 작업 스케줄러로 "아무도 안 보는 새벽"에 돌기 때문에, 잘못되면 며칠 뒤에야 알게 됩니다. 그래서 무인 실행에서
# 특히 위험한 부분을 실제 데이터 폴더가 아닌 임시 폴더에서 확인합니다:
#   1. 새 공고 판별 - 이미 있는 공고를 또 넣거나(청크 중복), 같은 공고가 목록에 두 번 나와 두 번 처리되지 않는지
#   2. 저장 - 네 산출물(metadata/extracted/chunks/embedded)의 개수·순서가 서로 맞게 덧붙는지 (test_data_pipeline.py 기준)
#   3. 잠금 - 동시에 두 번 돌지 않는지, 비정상 종료로 남은 오래된 잠금은 무시하는지
#   4. 실행 기록 - 성공/실패가 남고 오래된 기록은 잘리는지

import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sync"))

import sync_bizinfo as sync  # noqa: E402

results: list[bool] = []


def check(name: str, ok: bool, detail=None) -> None:
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok or detail is None else f"  ({detail})"))


def use_temp_paths(root: Path) -> None:
    """모듈이 쓰는 파일 경로를 전부 임시 폴더로 돌림 - 실제 data/는 절대 건드리지 않게."""
    (root / "raw").mkdir()
    (root / "processed").mkdir()
    sync.METADATA_CSV = root / "raw" / "metadata.csv"
    sync.EXTRACTED_JSON = root / "processed" / "extracted.json"
    sync.CHUNKS_JSON = root / "processed" / "chunks.json"
    sync.EMBEDDED_JSON = root / "processed" / "chunks_embedded.json"
    sync.STATE_PATH = root / "sync_state.json"
    sync.LOCK_PATH = root / ".sync.lock"


FIELDS = ["region", "pblancId", "title", "institution", "period", "target", "category", "source_url", "file_path", "downloaded"]


def meta(pid: str) -> dict:
    return {f: "" for f in FIELDS} | {"region": "서울", "pblancId": pid, "title": f"공고 {pid}", "downloaded": "True"}


def chunk(pid: str, i: int) -> dict:
    return {"chunk_id": f"{pid}_{i}", "program_id": pid, "chunk_text": f"{pid} 본문 {i}", "embedding": [0.1, 0.2]}


def test_find_new() -> None:
    print("── 1. 새 공고 판별 ──")
    listing = [{"pblancId": "A"}, {"pblancId": "B"}, {"pblancId": "B"}, {"pblancId": "C"}, {"pblancId": ""}]
    new = sync.find_new(listing, known_ids={"A"})
    check("이미 있는 공고(A)는 제외", "A" not in [i["pblancId"] for i in new])
    check("목록에 두 번 나온 공고(B)는 한 번만", [i["pblancId"] for i in new] == ["B", "C"], [i["pblancId"] for i in new])
    check("ID가 빈 항목은 무시", all(i["pblancId"] for i in new))


def test_persist(root: Path) -> None:
    print("\n── 2. 저장 (네 산출물 덧붙이기) ──")
    sync.METADATA_CSV.write_text(sync._csv_text(FIELDS, [meta("OLD")]), encoding="utf-8")
    sync.EXTRACTED_JSON.write_text(json.dumps([{"pblancId": "OLD", "text": "옛 본문"}]), encoding="utf-8")
    sync.CHUNKS_JSON.write_text(json.dumps([{**chunk("OLD", 0), "embedding": None}]), encoding="utf-8")
    sync.EMBEDDED_JSON.write_text(json.dumps([chunk("OLD", 0)]), encoding="utf-8")

    fields, rows = sync.load_metadata()
    new_results = [
        {"meta": meta("N1"), "doc": {"pblancId": "N1", "text": "새 본문"}, "chunks": [chunk("N1", 0), chunk("N1", 1)]},
        {"meta": meta("IMG"), "doc": None, "chunks": []},                                   # 이미지 공고: 메타만
        {"meta": meta("SCAN"), "doc": {"pblancId": "SCAN", "text": " "}, "chunks": []},     # 스캔본: 추출은 됐지만 청크 0
    ]
    sync.persist(fields, rows, new_results)

    _, rows_after = sync.load_metadata()
    extracted = json.loads(sync.EXTRACTED_JSON.read_text(encoding="utf-8"))
    chunks = json.loads(sync.CHUNKS_JSON.read_text(encoding="utf-8"))
    embedded = json.loads(sync.EMBEDDED_JSON.read_text(encoding="utf-8"))
    check("metadata: 기존 1 + 새 3 (이미지·스캔본도 '처리함'으로 기록 -> 매일 다시 받지 않음)",
          [r["pblancId"] for r in rows_after] == ["OLD", "N1", "IMG", "SCAN"], [r["pblancId"] for r in rows_after])
    check("extracted: 추출된 문서만 덧붙음 (이미지 제외)", [d["pblancId"] for d in extracted] == ["OLD", "N1", "SCAN"])
    check("chunks/embedded: 같은 개수·같은 순서", [c["chunk_id"] for c in chunks] == [c["chunk_id"] for c in embedded]
          == ["OLD_0", "N1_0", "N1_1"])
    check("chunks.json은 벡터 없이(배치 chunker 출력과 같은 모양), embedded에는 벡터",
          chunks[-1]["embedding"] is None and embedded[-1]["embedding"] == [0.1, 0.2])
    check("원자적 저장 후 임시 파일이 남지 않음", not list(root.rglob("*.tmp")), list(root.rglob("*.tmp")))
    check("CSV에 BOM(utf-8-sig) 유지 - 엑셀 한글 깨짐 방지", sync.METADATA_CSV.read_bytes().startswith(b"\xef\xbb\xbf"))
    raw = sync.METADATA_CSV.read_bytes()
    # 헤더 1줄 + 행 4줄 = 줄바꿈 5개. Windows 줄바꿈 변환이 끼면 csv의 "\r\n"이 "\r\r\n"으로 깨짐
    check("CSV 줄바꿈이 \\r\\r\\n으로 깨지지 않음 (Windows 줄바꿈 변환)", b"\r\r\n" not in raw and raw.count(b"\r\n") == 5,
          raw.count(b"\r\r\n"))


def test_lock() -> None:
    print("\n── 3. 잠금 ──")
    sync.acquire_lock()
    try:
        sync.acquire_lock()
        check("실행 중이면 두 번째 실행은 거절", False, "예외 없음")
    except sync.SyncLocked:
        check("실행 중이면 두 번째 실행은 거절", True)
    sync.release_lock()
    check("끝나면 잠금 해제", not sync.LOCK_PATH.exists())

    sync.LOCK_PATH.write_text("12345", encoding="utf-8")
    old = time.time() - sync.STALE_LOCK_SEC - 60
    os.utime(sync.LOCK_PATH, (old, old))
    try:
        sync.acquire_lock()
        check("비정상 종료로 남은 오래된 잠금은 무시하고 진행", True)
    except sync.SyncLocked:
        check("비정상 종료로 남은 오래된 잠금은 무시하고 진행", False)
    sync.release_lock()


def test_state() -> None:
    print("\n── 4. 실행 기록 ──")
    sync.save_run({"status": "success", "finished_at": "2026-09-29T06:00:00", "added_programs": 3})
    sync.save_run({"status": "failed", "error": "ConnectionError"})
    state = json.loads(sync.STATE_PATH.read_text(encoding="utf-8"))
    check("실패해도 last_success는 마지막 성공을 유지", state["last_success"]["added_programs"] == 3 and state["last_run"]["status"] == "failed")
    for i in range(sync.STATE_HISTORY + 5):
        sync.save_run({"status": "success", "n": i})
    state = json.loads(sync.STATE_PATH.read_text(encoding="utf-8"))
    check(f"기록은 최근 {sync.STATE_HISTORY}개만", len(state["runs"]) == sync.STATE_HISTORY and state["runs"][0]["n"] == sync.STATE_HISTORY + 4)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        use_temp_paths(root)
        test_find_new()
        test_persist(root)
        test_lock()
        test_state()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

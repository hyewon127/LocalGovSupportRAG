# [WBS 10 준비 / 2026-09-29] 기업마당 공고 자동 갱신 (증분 동기화)
#
# 실행 (OpenSearch 실행 중, 프로젝트 루트에서):
#   .venv\Scripts\python.exe src\sync\sync_bizinfo.py            # 새 공고 전부 반영
#   .venv\Scripts\python.exe src\sync\sync_bizinfo.py --dry-run  # 새 공고 목록만 보고 아무것도 바꾸지 않음
#   .venv\Scripts\python.exe src\sync\sync_bizinfo.py --limit 5  # 새 공고 5건만 (처음 시험할 때)
# 매일 자동 실행: scripts\register_sync_task.ps1 (Windows 작업 스케줄러 = 리눅스의 crontab 역할)
#
# [왜 만들었나]
#   WBS 2~4는 8월에 기업마당 API를 "한 번" 불러 574건을 받고 끝나는 배치였습니다. 9/29에 API를 다시 보니 서울 공고
#   517건 중 첫 페이지 100건이 전부 우리가 모르는 새 공고였습니다 - 지원사업 공고는 2~4주 단위로 올라오고 마감되므로,
#   한 번 받은 데이터는 한 달이면 대부분 "이미 끝난 사업"이 됩니다. 챗봇이 쓸모 있으려면 데이터가 계속 새로워야 합니다.
#
# [동작 - 배치 파이프라인(WBS 2~4)의 함수를 그대로 재사용하고, "새 공고만" 흘려보냄]
#   1. 목록 조회: API에서 현재 공고 목록 전체(메타데이터만, 몇 초)를 받음
#   2. 새 공고 찾기: data/raw/metadata.csv에 없는 공고 ID = 새 공고
#   3. 새 공고만: 첨부 다운로드 -> 텍스트 추출(extractor) -> 청킹(chunker) -> 임베딩(embedder) -> 색인(bulk)
#   4. 저장: 색인이 성공한 뒤에만 metadata.csv / extracted.json / chunks.json / chunks_embedded.json 뒤에 덧붙임
#
# [안전장치 - 스케줄러로 무인 실행되므로 "중간에 죽어도 다음 실행이 알아서 복구"가 목표]
#   - 순서: 메모리에서 전부 만든 뒤 "색인 -> 파일" 순으로 반영. 색인에 실패하면 파일을 안 바꾸므로 그 공고들은 다음 실행에서
#     다시 "새 공고"로 잡혀 재시도됩니다. 색인은 _id=chunk_id라 두 번 넣어도 덮어쓰기(멱등)라서 중복이 생기지 않습니다.
#   - 파일은 임시 파일에 다 쓴 뒤 교체(os.replace)해서, 쓰다가 죽어도 반쯤 쓰인 JSON이 남지 않게 합니다.
#   - 잠금 파일(data/.sync.lock)로 두 개가 동시에 돌지 않게 합니다 (스케줄 실행 + 수동 실행이 겹치는 경우).
#   - 실행 기록은 data/sync_state.json에 남고, API의 GET /sync/status와 웹 화면 헤더("최근 갱신")가 이걸 읽습니다.
#
# [하지 않는 것 - 의도적으로 뺌]
#   - 마감되거나 목록에서 사라진 공고를 지우지 않습니다. 평가셋(data/eval)이 8월 공고로 라벨링돼 있어서 지우면 평가를 재현할 수
#     없고, 마감 공고도 "작년에 이런 사업이 있었다"는 참고가 됩니다. 대신 웹 화면 카드에 "마감" 표시를 합니다(web/chat-format.js).
#   - 이미 있는 공고의 내용 변경(정정 공고)은 반영하지 않습니다. 기업마당은 정정 시 보통 새 공고 ID로 다시 올립니다.

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _module_dir in ("crawler", "preprocessing", "indexing"):
    _path = str(_PROJECT_ROOT / "src" / _module_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fetch_bizinfo  # noqa: E402  src/crawler - API 호출/파싱/다운로드
from chunker import build_chunk_record, chunk_sentences, split_sentences  # noqa: E402  src/preprocessing
from config import CHUNKS_JSON, EMBEDDED_JSON, INDEX_NAME, check_connection, get_client  # noqa: E402  src/indexing
from extractor import classify_format, extract_hwp_ole, extract_hwpx, extract_pdf  # noqa: E402

DATA_DIR = _PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "raw" / "metadata.csv"
EXTRACTED_JSON = DATA_DIR / "processed" / "extracted.json"
STATE_PATH = DATA_DIR / "sync_state.json"
LOCK_PATH = DATA_DIR / ".sync.lock"
LOG_PATH = _PROJECT_ROOT / "logs" / "sync.log"

# 잠금 파일이 이보다 오래됐으면 이전 실행이 비정상 종료(전원 차단 등)된 것으로 보고 무시. 전체 재수집도 1시간 안쪽이라 넉넉히 3시간.
STALE_LOCK_SEC = 3 * 3600
STATE_HISTORY = 30  # sync_state.json에 남길 최근 실행 기록 수

log = logging.getLogger("sync")


def setup_logging() -> None:
    # 스케줄러로 돌면 콘솔을 볼 사람이 없으므로 파일(logs/sync.log)에도 남김. 콘솔은 수동 실행할 때 보라고 같이 둠.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


# ── 0. 잠금 / 기록 ───────────────────────────────────────────────────
class SyncLocked(Exception):
    pass


def acquire_lock() -> None:
    if LOCK_PATH.exists():
        age = time.time() - LOCK_PATH.stat().st_mtime
        if age < STALE_LOCK_SEC:
            raise SyncLocked(f"다른 동기화가 실행 중입니다 ({LOCK_PATH.name}, {age / 60:.0f}분 전 시작)")
        log.warning("오래된 잠금 파일(%.0f분)을 무시합니다 - 이전 실행이 비정상 종료된 것으로 봄", age / 60)
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"runs": []}


def save_run(run: dict) -> None:
    state = load_state()
    state["last_run"] = run
    if run["status"] == "success":
        state["last_success"] = run
    state["runs"] = ([run] + state.get("runs", []))[:STATE_HISTORY]
    write_atomic(STATE_PATH, json.dumps(state, ensure_ascii=False, indent=2))


def write_atomic(path: Path, text: str) -> None:
    """임시 파일에 다 쓴 뒤 한 번에 교체 - 쓰는 도중 죽어도 원본은 온전함 (os.replace는 같은 드라이브 안에서 원자적)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    # newline="": 줄바꿈을 그대로 씀. 기본값이면 Windows가 LF를 CRLF로 바꿔서, csv가 이미 CRLF로 끝낸 줄이
    # CR CR LF가 됨 (이 프로젝트의 실행 스크립트를 만들 때 실제로 겪은 문제 - tests/test_sync.py에서 검사)
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


# ── 1~2. 목록 조회 / 새 공고 찾기 ─────────────────────────────────────
def fetch_listing(regions: list[str]) -> list[dict]:
    """API의 현재 공고 목록(메타데이터만). fetch_bizinfo.collect_region()과 같은 페이지 순회지만 다운로드는 안 함."""
    listing = []
    for hashtag in regions:
        page, collected = 1, 0
        while True:
            items, total = fetch_bizinfo.parse_items(fetch_bizinfo.fetch_page(hashtag, page))
            for item in items:
                item["_region"] = hashtag
            listing.extend(items)
            collected += len(items)
            if not items or collected >= total or len(items) < fetch_bizinfo.PAGE_UNIT:
                break
            page += 1
            time.sleep(0.3)  # 배치 수집기와 같은 간격 - 공공 API에 연속 요청을 몰아 보내지 않음
        log.info("[목록] %s: %d건", hashtag, collected)
    return listing


def load_metadata() -> tuple[list[str], list[dict]]:
    with open(METADATA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def find_new(listing: list[dict], known_ids: set[str]) -> list[dict]:
    """목록 중 처음 보는 공고. 같은 공고가 여러 지역 해시태그에 걸려 목록에 두 번 나올 수 있어서 ID로 한 번만."""
    seen, new = set(), []
    for item in listing:
        pid = item["pblancId"]
        if pid and pid not in known_ids and pid not in seen:
            seen.add(pid)
            new.append(item)
    return new


# ── 3. 새 공고 처리 (다운로드 -> 추출 -> 청킹) ─────────────────────────
def process_item(item: dict) -> dict | None:
    """
    공고 1건 -> {"meta": metadata.csv 행, "doc": extracted.json 원소 또는 None, "chunks": 청크 레코드들}
    다운로드 실패면 None - metadata에 안 남기므로 다음 실행에서 다시 시도됨 (일시적인 네트워크 오류 대비).
    형식 미지원(이미지)·추출 실패는 metadata에는 남겨서(배치 수집기와 같은 기준) 매번 다시 받지 않게 함.
    """
    hashtag = item["_region"]
    ext = Path(item["printFileNm"]).suffix or ".pdf"
    safe_id = re.sub(r"[^\w\-]", "_", item["pblancId"])  # 파일명 규칙은 fetch_bizinfo.collect_region()과 같게
    save_path = fetch_bizinfo.RAW_DIR / f"{hashtag}_{safe_id}{ext}"
    if not fetch_bizinfo.download_file(item["printFlpthNm"], save_path):
        return None

    meta = {
        "region": hashtag, "pblancId": item["pblancId"], "title": item["pblancNm"],
        "institution": item["jrsdInsttNm"], "period": item["reqstBeginEndDe"], "target": item["trgetNm"],
        "category": item["pldirSportRealmLclasCodeNm"], "source_url": item["pblancUrl"],
        "file_path": str(save_path), "downloaded": "True",  # csv에서 읽은 값과 같은 문자열 (extractor가 "True"로 비교)
    }
    fmt = classify_format(save_path)  # 확장자가 아니라 파일 앞부분 바이트로 판별 (.pdf로 받았는데 HWP인 경우가 실제로 있음)
    if fmt == "unsupported":
        return {"meta": meta, "doc": None, "chunks": [], "note": f"미지원 형식 {save_path.suffix}"}
    try:
        extractor = {"pdf": extract_pdf, "hwp_ole": extract_hwp_ole}.get(fmt, extract_hwpx)
        doc = extractor(save_path)
    except Exception as e:  # 깨진 파일 하나 때문에 전체 동기화가 멈추면 안 됨 (extractor.process_all과 같은 방침)
        return {"meta": meta, "doc": None, "chunks": [], "note": f"추출 실패 {e}"}
    doc.update({"pblancId": item["pblancId"], "title": item["pblancNm"], "region": hashtag})

    chunks = []
    if doc["text"].strip():  # 스캔본처럼 글자가 없는 문서는 청크 0개 (chunker.process_all과 같은 기준)
        for index, chunk in enumerate(chunk_sentences(split_sentences(doc["text"]))):
            chunks.append(build_chunk_record(doc=doc, meta=meta, chunk=chunk, index=index))
    return {"meta": meta, "doc": doc, "chunks": chunks, "note": None if chunks else "텍스트 없음"}


# ── 4. 임베딩 / 색인 / 저장 ──────────────────────────────────────────
def embed_records(records: list[dict]) -> None:
    from embedder import BATCH_SIZE, embed_batch  # 모델 로딩이 무거워서 새 청크가 있을 때만 import
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        for record, vector in zip(batch, embed_batch([r["chunk_text"] for r in batch])):
            record["embedding"] = vector


def index_records(client, records: list[dict]) -> None:
    from bulk_indexer import generate_actions
    from opensearchpy import helpers
    success, errors = helpers.bulk(client, generate_actions(records), raise_on_error=False)
    if errors:
        # 일부라도 실패하면 파일을 갱신하지 않고 멈춤 -> 다음 실행에서 이번 공고들 전체를 다시 시도 (색인은 덮어쓰기라 안전)
        raise RuntimeError(f"색인 실패 {len(errors)}건 (성공 {success}건). 첫 오류: {errors[0]}")
    client.indices.refresh(index=INDEX_NAME)  # 바로 검색되게 (기본은 1초 주기 반영)


def persist(fieldnames: list[str], old_rows: list[dict], results: list[dict]) -> None:
    """배치 파이프라인 산출물 4개 뒤에 새 결과를 덧붙임. 네 파일의 순서·개수가 서로 맞아야 tests/test_data_pipeline.py가 통과."""
    new_metas = [r["meta"] for r in results]
    new_docs = [r["doc"] for r in results if r["doc"] is not None]
    new_chunks = [c for r in results for c in r["chunks"]]

    rows_text = _csv_text(fieldnames, old_rows + new_metas)
    extracted = json.loads(EXTRACTED_JSON.read_text(encoding="utf-8")) + new_docs
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    embedded = json.loads(EMBEDDED_JSON.read_text(encoding="utf-8"))
    # chunks.json은 embedding=None 상태로 저장 (배치 chunker.py 출력과 같은 모양)
    chunks += [{**c, "embedding": None} for c in new_chunks]
    embedded += new_chunks

    # 전부 메모리에서 만든 뒤 한꺼번에 교체. 형식(들여쓰기 여부)도 각 배치 스크립트의 출력과 같게 맞춤.
    write_atomic(EXTRACTED_JSON, json.dumps(extracted, ensure_ascii=False, indent=2))
    write_atomic(CHUNKS_JSON, json.dumps(chunks, ensure_ascii=False, indent=2))
    write_atomic(EMBEDDED_JSON, json.dumps(embedded, ensure_ascii=False))
    # metadata.csv를 가장 마지막에: 이게 "이미 처리한 공고" 목록이라, 앞 파일 저장 중 죽으면 다음 실행이 다시 처리하게 됨
    write_atomic(METADATA_CSV, rows_text)


def _csv_text(fieldnames: list[str], rows: list[dict]) -> str:
    import io
    buffer = io.StringIO()
    buffer.write("﻿")  # utf-8-sig (엑셀에서 한글이 안 깨지게 - 배치 수집기와 같은 인코딩)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


# ── 실행 ────────────────────────────────────────────────────────────
def run(limit: int | None = None, dry_run: bool = False) -> dict:
    started = datetime.now()
    run_info = {"started_at": started.isoformat(timespec="seconds"), "status": "running", "dry_run": dry_run}
    client = get_client()
    if not check_connection(client):
        raise ConnectionError("OpenSearch에 연결할 수 없습니다 - 스케줄 시간에 OpenSearch가 켜져 있어야 합니다")

    listing = fetch_listing(fetch_bizinfo.TARGET_REGIONS)
    fieldnames, rows = load_metadata()
    new_items = find_new(listing, {r["pblancId"] for r in rows})
    run_info.update({"listed": len(listing), "known": len(rows), "new_found": len(new_items)})
    log.info("[비교] API 목록 %d건 / 이미 있음 %d건 / 새 공고 %d건", len(listing), len(rows), len(new_items))
    if limit is not None:
        new_items = new_items[:limit]
    if dry_run or not new_items:
        for item in new_items[:20]:
            log.info("  새 공고: %s %s (%s)", item["pblancId"], item["pblancNm"][:40], item["reqstBeginEndDe"])
        return {**run_info, "status": "success", "added_programs": 0, "added_chunks": 0}

    results, download_failed = [], 0
    for i, item in enumerate(new_items, 1):
        result = process_item(item)
        if result is None:
            download_failed += 1
            log.warning("  [%d/%d] 다운로드 실패 - 다음 실행에서 재시도: %s", i, len(new_items), item["pblancId"])
            continue
        results.append(result)
        log.info("  [%d/%d] %s 청크 %d개%s", i, len(new_items), item["pblancNm"][:30], len(result["chunks"]),
                 f" ({result['note']})" if result["note"] else "")

    new_chunks = [c for r in results for c in r["chunks"]]
    if new_chunks:
        log.info("[임베딩] 청크 %d개", len(new_chunks))
        embed_records(new_chunks)
        log.info("[색인] %s에 %d개", INDEX_NAME, len(new_chunks))
        index_records(client, new_chunks)
    persist(fieldnames, rows, results)
    index_total = client.count(index=INDEX_NAME)["count"]
    return {**run_info, "status": "success",
            "added_programs": len(results), "added_chunks": len(new_chunks),
            "download_failed": download_failed,
            "no_text_or_unsupported": sum(1 for r in results if not r["chunks"]),
            "index_total": index_total}


def main() -> int:
    parser = argparse.ArgumentParser(description="기업마당 새 공고를 찾아 색인까지 반영 (증분 동기화)")
    parser.add_argument("--limit", type=int, help="새 공고 중 앞에서 N건만 처리 (시험용)")
    parser.add_argument("--dry-run", action="store_true", help="새 공고 목록만 보여주고 아무것도 바꾸지 않음")
    args = parser.parse_args()
    setup_logging()

    started = time.time()
    try:
        acquire_lock()
    except SyncLocked as e:
        log.warning(str(e))
        return 2
    try:
        run_info = run(limit=args.limit, dry_run=args.dry_run)
    except Exception as e:  # 무인 실행이라 원인을 기록에 남기고 실패 코드로 끝냄 (작업 스케줄러 "마지막 실행 결과"에 표시됨)
        log.exception("동기화 실패")
        run_info = {"started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
                    "status": "failed", "error": f"{type(e).__name__}: {e}"}
    finally:
        release_lock()
    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_info["duration_sec"] = round(time.time() - started, 1)
    if not args.dry_run:
        save_run(run_info)
    log.info("[결과] %s", json.dumps(run_info, ensure_ascii=False))
    return 0 if run_info["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

# [WBS 8.2] 데이터 파이프라인 정합성 테스트 - 수집 -> 추출 -> 청킹 -> 임베딩 -> 색인 단계의 산출물이 서로 맞물리는지
#
# 실행 (OpenSearch 실행 중, 프로젝트 루트에서, 약 30초 - 76MB 임베딩 파일 로딩):
#   .venv\Scripts\python.exe tests\test_data_pipeline.py
#
# [왜 "다시 돌리기"가 아니라 "산출물 대조"인가]
#   크롤러는 기업마당 API 키와 네트워크가 필요하고, 추출(extractor.py)은 574개 파일에 10분 이상 걸립니다.
#   통합 테스트를 돌릴 때마다 이걸 다시 할 수는 없으므로, 각 단계가 남긴 파일(와 인덱스)을 서로 대조해서
#   "앞 단계의 출력이 빠짐없이, 변형 없이 다음 단계의 입력이 되었는가"를 확인합니다.
#   예) 청크를 다시 만들고 임베딩을 안 다시 돌리면 chunk_text와 벡터가 서로 다른 텍스트를 가리키게 되는데,
#       이런 "단계 간 어긋남"은 검색 결과만 봐서는 알아채기 어렵습니다.
#
# [기준값] 2026-09-22 산출물 기준 (메타데이터 574 -> 추출 533 -> 텍스트 있는 문서 491 -> 청크 3,884 -> 색인 3,884).
#   개수를 하드코딩하지 않고 "앞 단계에서 계산한 값과 같은가"로 비교하므로, 데이터를 다시 수집해도 테스트는 그대로 쓸 수 있습니다.

import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "indexing"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "preprocessing"))

from chunker import CHUNK_MAX_TOKENS  # noqa: E402
from config import CHUNKS_JSON, EMBEDDED_JSON, EMBEDDING_DIM, INDEX_NAME, PROCESSED_DIR, get_client  # noqa: E402

METADATA_CSV = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
EXTRACTED_JSON = PROCESSED_DIR / "extracted.json"
results: list[tuple[str, bool]] = []


def check(name: str, condition, detail="") -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""))


def main() -> int:
    for path in (METADATA_CSV, EXTRACTED_JSON, CHUNKS_JSON, EMBEDDED_JSON):
        if not path.exists():
            print(f"[중단] 산출물이 없습니다: {path} - 파이프라인(crawler -> preprocessing -> indexing)을 먼저 실행하세요.")
            return 1

    # ── 1. 수집 (fetch_bizinfo.py -> data/raw) ──
    print("── 1. 수집 ──")
    rows = list(csv.DictReader(METADATA_CSV.open(encoding="utf-8-sig")))
    meta_ids = {r["pblancId"] for r in rows}
    check("메타데이터: 공고 ID 중복 없음", len(meta_ids) == len(rows), f"{len(rows)}행")
    missing = [r["file_path"] for r in rows if r["downloaded"] == "True" and not Path(r["file_path"]).exists()]
    check("메타데이터: 다운로드됐다고 기록된 파일이 실제로 존재", not missing, f"누락 {len(missing)}건")

    # ── 2. 추출 (extractor.py -> extracted.json) ──
    print("── 2. 추출 ──")
    extracted = json.loads(EXTRACTED_JSON.read_text(encoding="utf-8"))
    ext_ids = {e["pblancId"] for e in extracted}
    check("추출: 모든 문서가 수집 메타데이터에 있는 공고", ext_ids <= meta_ids, f"{len(extracted)}건")
    with_text = {e["pblancId"] for e in extracted if e["text"].strip()}
    print(f"   참고: 추출 {len(extracted)}건 중 텍스트가 빈 문서 {len(extracted) - len(with_text)}건 "
          f"(스캔본 등 - OCR 미지원이라 청크가 생기지 않음)")

    # ── 3. 청킹 (chunker.py -> chunks.json) ──
    print("── 3. 청킹 ──")
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_programs = {c["program_id"] for c in chunks}
    check("청크: chunk_id 중복 없음 (색인 시 _id로 쓰여서 중복이면 덮어써짐)", len(chunk_ids) == len(set(chunk_ids)), f"{len(chunks)}개")
    check("청크: 텍스트가 있는 추출 문서는 전부 청크가 됨 (누락 없음)", chunk_programs == with_text,
          f"청크 사업 {len(chunk_programs)} / 텍스트 있는 문서 {len(with_text)}")
    required = ["program_id", "program_name", "region_name", "category", "chunk_text", "source_url"]
    empty_required = {k: sum(1 for c in chunks if not c.get(k)) for k in required}
    check("청크: 필수 필드(검색·필터·출처 인용에 쓰임) 빈 값 없음", not any(empty_required.values()), empty_required)
    over = sum(1 for c in chunks if c["token_count"] > CHUNK_MAX_TOKENS)
    check(f"청크: 토큰 수 상한({CHUNK_MAX_TOKENS}) 준수", over == 0, f"초과 {over}개")
    bad_dates = [c["chunk_id"] for c in chunks if c["apply_start"] and c["apply_end"]
                 and date.fromisoformat(c["apply_start"]) > date.fromisoformat(c["apply_end"])]
    check("청크: 접수 시작일 <= 종료일", not bad_dates, bad_dates[:3])
    check("청크: 출처 URL이 자기 공고 ID를 가리킴 (틀린 링크 인용 방지)",
          all(c["program_id"] in c["source_url"] for c in chunks))

    # ── 4. 임베딩 (embedder.py -> chunks_embedded.json) ──
    print("── 4. 임베딩 ──")
    embedded = json.loads(EMBEDDED_JSON.read_text(encoding="utf-8"))
    check("임베딩: 청크와 같은 개수·같은 순서", [e["chunk_id"] for e in embedded] == chunk_ids, f"{len(embedded)}개")
    check("임베딩: 벡터가 만들어진 텍스트 = 현재 청크 텍스트 (청킹만 다시 하고 임베딩을 안 돌린 상태 검출)",
          all(e["chunk_text"] == c["chunk_text"] for e, c in zip(embedded, chunks)))
    dims = {len(e["embedding"]) for e in embedded}
    check(f"임베딩: 전부 {EMBEDDING_DIM}차원 (config.EMBEDDING_DIM)", dims == {EMBEDDING_DIM}, dims)
    check("임베딩: NaN/무한대 없음", all(math.isfinite(v) for e in embedded for v in e["embedding"]))

    # ── 5. 색인 (bulk_indexer.py -> OpenSearch) ──
    print("── 5. 색인 ──")
    client = get_client()
    if not client.ping():
        check("OpenSearch 연결", False, "꺼져 있음")
        return 1
    count = client.count(index=INDEX_NAME)["count"]
    check("색인: 문서 수 = 임베딩 파일 청크 수", count == len(embedded), f"인덱스 {count} / 파일 {len(embedded)}")
    agg = client.search(index=INDEX_NAME, body={"size": 0, "aggs": {"p": {"terms": {"field": "program_id", "size": 5000}}}})
    index_programs = {b["key"] for b in agg["aggregations"]["p"]["buckets"]}
    check("색인: 사업 목록 = 청크 파일의 사업 목록", index_programs == chunk_programs, f"{len(index_programs)}개")
    mapping_dim = client.indices.get_mapping(index=INDEX_NAME)[INDEX_NAME]["mappings"]["properties"]["embedding"]["dimension"]
    check("색인: 매핑의 벡터 차원 = 임베딩 차원", mapping_dim == EMBEDDING_DIM, mapping_dim)

    # 표본 대조: 전체를 다 비교하면 느리므로 간격을 두고 30개를 뽑아(매번 같은 표본) 원문 파일과 비교
    sample = embedded[:: max(1, len(embedded) // 30)][:30]
    docs = client.mget(index=INDEX_NAME, body={"ids": [e["chunk_id"] for e in sample]}, _source=["chunk_text", "program_id"])["docs"]
    mismatched = [d["_id"] for d, e in zip(docs, sample)
                  if not d.get("found") or d["_source"]["chunk_text"] != e["chunk_text"]]
    check("색인: 표본 30개의 본문이 파일과 일치 (_id = chunk_id)", not mismatched, mismatched[:3])

    # 자기 검색: 저장된 벡터로 kNN 검색하면 자기 자신(또는 본문이 똑같은 중복 공고)이 1등이어야 함.
    # 벡터가 다른 문서에 잘못 붙어 색인됐다면 여기서 드러납니다.
    wrong_top = []
    for e in sample[:5]:
        top = client.search(index=INDEX_NAME, body={"size": 1, "_source": ["chunk_text"],
                                                    "query": {"knn": {"embedding": {"vector": e["embedding"], "k": 1}}}})
        hit = top["hits"]["hits"][0]
        if hit["_id"] != e["chunk_id"] and hit["_source"]["chunk_text"] != e["chunk_text"]:
            wrong_top.append(e["chunk_id"])
    check("색인: 저장된 벡터로 검색하면 자기 자신이 1등 (벡터-문서 짝 검증, 표본 5개)", not wrong_top, wrong_top)

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

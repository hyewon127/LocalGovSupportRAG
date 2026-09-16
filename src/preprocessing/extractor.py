# %%
#라이브러리 선언
import csv
import json
from collections import Counter  # 건너뛴 파일들을 확장자별로 세기 위함 (fetch_bizinfo.py에서 쓴 것과 같은 패턴)
from pathlib import Path

import pdfplumber
# pdfplumber를 씀: 오늘 PyMuPDF랑 비교해봤을 때 표 구조(extract_tables)를 훨씬 잘 살려줘서,
# "표/본문 분리"가 필요한 요구사항(SFR-002)에는 이쪽이 더 맞음. 속도는 느리지만 574건 정도는 감내 가능.

# %%
#0. 경로 설정
#파일 ROOT 확인(parents 상위 폴더로 몇번 가는지 확인)
#__file__ 은 "python extractor.py"처럼 일반 스크립트로 실행할 때만 정의되고,
#주피터/셀 단위 실행(노트북)에서는 NameError가 남 -> 그럴 땐 예외로 잡아서
#"노트북 파일이 있는 폴더 = cwd"라는 Jupyter 표준 동작을 이용해 대신 계산함
#(fetch_bizinfo.ipynb에서 썼던 것과 같은 방식 - 이 파일 위치 기준 상위 1번 = 프로젝트 루트)
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    #path(__file__) 이 안먹히는 경우 Path.cwd()으로 설정
    PROJECT_ROOT = Path.cwd().resolve().parents[1]
#data row 있는 경로
RAW_DIR = PROJECT_ROOT / "data" / "raw"
#청킹 파일 담을 경로
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
#경로가 없는 경우 생성
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
# fetch_bizinfo.py가 만들어둔 수집 목록
METADATA_CSV = RAW_DIR / "metadata.csv"

# %%
#1. 파일 확장자 확인(현재로는 pdf로만 청킹 진행/hwp+hwpx+img는 추후 구현)
def is_supported_format(file_path: Path) -> bool:
    return file_path.suffix.lower() == ".pdf"

# %%
#2. PDF 텍스트 + 표 추출
def extract_pdf(pdf_path: Path) -> dict:
    #페이지별 텍스트를 모아둘 리스트
    pages_text = []
    #페이지 상관없이 찾은 표를 전부 모아둘 리스트 (표 하나 = 행x열 리스트)
    all_tables = []

    #with 문: 파일을 열고 블록이 끝나면 자동으로 닫아줌 (파일 핸들 leak 방지)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            #텍스트가 없는 페이지(이미지만 있는 스캔본 등) 대비 기본값 ""
            text = page.extract_text() or ""
            pages_text.append(text)

            #extract_tables(): 좌표 간격을 분석해서 표처럼 보이는 영역을 뽑아냄
            for table in page.extract_tables():
                all_tables.append(table)

    full_text = "\n".join(pages_text)

    return {
        "file_path": str(pdf_path),
        "page_count": len(pages_text),
        "text": full_text,
        "char_count": len(full_text),
        "table_count": len(all_tables),
        "tables": all_tables,
    }

# %%
#3. metadata.csv 읽어서 수집된 공고 목록 불러오기
def load_metadata_rows() -> list[dict]:
    with open(METADATA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        #아까 만들어둔 list 에 목록 담기
        return list(reader)

# %%
#4. 전체 배치 처리: metadata.csv 순회하며 PDF만 추출해서 결과를 하나의 JSON으로 저장
def process_all() -> None:
    rows = load_metadata_rows()
    results = []

    #각 케이스별로 몇 건씩인지 세어서 마지막에 요약 출력 (전수 확인 없이 감으로 넘어가지 않기 위함)
    skipped_not_downloaded = 0
    #건너뛴 파일을 확장자별로 세는 Counter -> 파일마다 한 줄씩 찍던 걸 없애고 마지막에 요약만 보여줌
    skipped_unsupported_by_ext = Counter()
    failed = 0

    #4-1. 청킹
    for row in rows:
        #다운로드 자체가 실패한 행은 처리 대상 아님
        if row["downloaded"] != "True":
            skipped_not_downloaded += 1
            continue

        file_path = Path(row["file_path"])

        if not is_supported_format(file_path):
            #파일마다 print 하지 않고 확장자별 개수만 누적 (220건이면 220줄 찍혀서 안 보임)
            skipped_unsupported_by_ext[file_path.suffix.lower()] += 1
            continue

        #pdf 가 아닌 파일들에 대한 처리
        try:
            extracted = extract_pdf(file_path)
        except Exception as e:
            #한 파일이 깨져있어도 전체 배치가 멈추면 안 되므로 예외를 잡고 계속 진행
            failed += 1
            print(f"[실패] {file_path.name} - {e}")
            continue

        #추출 결과에 원본 메타데이터(공고ID/제목/지역)를 같이 묶어둠 -> 나중에 청킹 단계에서 그대로 재사용
        extracted["pblancId"] = row["pblancId"]
        extracted["title"] = row["title"]
        extracted["region"] = row["region"]
        results.append(extracted)

    #결과를 JSON으로 저장
    out_path = PROCESSED_DIR / "extracted.json"
    with open(out_path, "w", encoding="utf-8") as f:
        #ensure_ascii=False: 한글이 유니코드 이스케이프(\uXXXX)로 안 바뀌고 그대로 저장되게 함
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_skipped_unsupported = sum(skipped_unsupported_by_ext.values())

    print(f"\n총 {len(rows)}건 중:")
    print(f"  추출 성공: {len(results)}건")
    print(f"  미다운로드라 건너뜀: {skipped_not_downloaded}건")
    print(f"  미지원 형식이라 건너뜀: {total_skipped_unsupported}건")
    #확장자별로 몇 건씩 건너뛰었는지 세부 내역 (most_common: 개수 많은 순으로 정렬)
    for ext, count in skipped_unsupported_by_ext.most_common():
        print(f"    - {ext}: {count}건")
    print(f"  추출 중 실패: {failed}건")
    print(f"저장 위치: {out_path}")

# %%
if __name__ == "__main__":
    process_all()

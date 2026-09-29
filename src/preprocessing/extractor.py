# %%
#라이브러리 선언
import csv
import json
import struct   # HWP 바이너리 레코드 헤더(4바이트 정수) 해석용
import zipfile  # .hwpx(=zip) 내부 파일을 열기 위함
import zlib     # .hwp 본문 스트림의 압축(raw deflate) 해제용
import xml.etree.ElementTree as ET  # .hwpx 내부 XML(문단 텍스트) 파싱용
from collections import Counter  # 건너뛴 파일들을 확장자별로 세기 위함 (fetch_bizinfo.py에서 쓴 것과 같은 패턴)
from pathlib import Path

import pdfplumber
# pdfplumber를 씀: 오늘 PyMuPDF랑 비교해봤을 때 표 구조(extract_tables)를 훨씬 잘 살려줘서,
# "표/본문 분리"가 필요한 요구사항(SFR-002)에는 이쪽이 더 맞음. 속도는 느리지만 574건 정도는 감내 가능.

try:
    import olefile  # 구버전 .hwp(OLE 컴파운드 파일)를 열기 위한 라이브러리. pip install olefile 필요
    HAS_OLEFILE = True
except ImportError:
    # 미설치 상태여도 PDF 처리는 그대로 돌아가야 하므로 방어적으로 처리
    # (extract_hwp_ole() 호출 시점에 명확한 에러 메시지로 알려줌)
    HAS_OLEFILE = False

# [2026-09-16 예외 기록] 아래 hwp/hwpx 관련 함수들(classify_format, extract_hwp_ole,
# extract_hwpx, _decode_hwp_para_text)은 원래 확인용/HWP·HWPX 텍스트 추출.py에서
# 학습용으로 먼저 만들고 실제 data/raw/ 219건 전체로 검증(181/181 성공)까지 마친 뒤,
# 하예님이 이번에도(chunker.py 때처럼) src/에 직접 반영해달라고 명시적으로 요청하셔서
# 제가 그대로 옮겨 적었습니다. 로직과 주석은 확인용 스크립트와 동일합니다.

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
#1. 파일 포맷 판별 - 확장자가 아니라 실제 내용(매직바이트)으로 확인
#
# [왜 확장자를 못 믿는가 - 실제로 확인된 문제] data/raw의 .hwpx 48개를 전수 확인해보니
# 4개는 zipfile로 안 열리고 첫 8바이트가 D0 CF 11 E0 A1 B1 1A E1 였음 - 이건 zip
# 시그니처가 아니라 "OLE 컴파운드 파일"(구버전 .hwp와 동일) 시그니처. 즉 이 4개는
# 확장자만 .hwpx일 뿐 실제로는 .hwp임. printFlpthNm이 .pdf인데 실제론 .hwp였던
# 문제와 같은 종류라, hwp/hwpx는 반드시 매직바이트로 실제 포맷을 확인하고 분기함.
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def classify_format(file_path: Path) -> str:
    """
    입력: 파일 경로
    출력: "pdf" | "hwp_ole" | "hwpx_zip" | "unsupported"
          (unsupported = png/jpg/docx 등 아직 처리 로직이 없는 포맷 - OCR은 후속 작업으로 보류)
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".hwp", ".hwpx"):
        with open(file_path, "rb") as f:
            head = f.read(8)
        if head == _OLE_SIGNATURE:
            return "hwp_ole"
        if head[:2] == b"PK":  # zip 파일은 항상 "PK"로 시작함 (zip 포맷 개발자 Phil Katz의 이니셜)
            return "hwpx_zip"
        return "unsupported"  # 둘 다 아니면 손상 파일 등 - 처리 못 함
    return "unsupported"

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
#2-1. 구버전 .hwp(OLE 컴파운드 파일) 텍스트 추출
#
# HWP 5.0 내부 구조: OLE 파일 안에 "BodyText/Section0", "Section1", ... 스트림이 있고,
# 각 스트림은 (보통) zlib으로 압축된 "레코드"의 나열임. 레코드 하나 = [4바이트 헤더][본문].
# 태그 번호 67번(HWPTAG_PARA_TEXT)이 "문단 텍스트" 레코드라서 그것만 골라 읽으면 본문이 나옴.
#
# 문단 텍스트 안에는 순수 글자 말고 표/그림/필드를 가리키는 제어문자도 섞여 있어서,
# 걸러내지 않고 그냥 디코딩하면 "捤獥 汤捯 氠瑢" 같은 깨진 문자가 섞여 나옴(실측 확인됨).
# HWP 스펙상 제어문자는 두 종류:
#   - "확장 제어문자"(표/그림/필드 등): 문자 1개 + 부가정보 7개 = 총 8개 코드유닛을 통째로 건너뜀
#   - "인라인 제어문자"(탭/줄바꿈 등): 문자 1개만 소비. 9(탭)/10(줄바꿈)/13(문단끝)은 개행으로 바꿔줌
_HWP_EXTENDED_CTRL = {1, 2, 3, 11, 12, 14, 15, 16, 17, 21, 22, 23}
_HWP_INLINE_CTRL = {4, 5, 6, 7, 8, 9, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}
_HWP_PARA_TEXT_TAG = 67  # HWPTAG_PARA_TEXT


def _decode_hwp_para_text(raw: bytes) -> str:
    """입력: HWPTAG_PARA_TEXT 레코드 원본 바이트(UTF-16LE) / 출력: 제어문자를 걸러낸 순수 텍스트"""
    units = struct.unpack(f"<{len(raw) // 2}H", raw[: len(raw) // 2 * 2])
    out = []
    i, n = 0, len(units)
    while i < n:
        ch = units[i]
        if ch in _HWP_EXTENDED_CTRL:
            i += 8  # 제어문자 자신 + 파라미터 7개를 통째로 건너뜀
        elif ch in _HWP_INLINE_CTRL:
            if ch in (9, 10, 13):
                out.append("\n")
            i += 1
        elif ch == 0:
            i += 1  # 패딩용 null
        elif 0xD800 <= ch <= 0xDFFF:
            # [2026-09-16 실제 발견된 버그] UTF-16 서로게이트(surrogate) 코드유닛임.
            # 정상적인 서로게이트 쌍(둘이 합쳐 이모지 등 BMP 밖 문자 하나를 이룸)이라면 문제 없지만,
            # 공고문 본문에 그런 문자가 나올 일은 거의 없고, 실제로는 위 제어문자 필터링이
            # 완벽하지 않아 레코드 경계가 살짝 어긋나면서 나온 쓰레기 값으로 추정됨(실측: 354개 PDF
            # + 181개 hwp/hwpx를 처리하다가 이 값이 껴서 json.dump()가
            # "UnicodeEncodeError: surrogates not allowed"로 죽는 걸 확인함).
            # 짝 없는 서로게이트는 UTF-8로 인코딩이 안 되는 값이라 그냥 버림 - 본문 손실은 미미하고,
            # 안 버리면 파이프라인 전체가 마지막(json.dump) 단계에서 통째로 죽음.
            i += 1
        else:
            out.append(chr(ch))
            i += 1
    return "".join(out)


def extract_hwp_ole(path: Path) -> dict:
    """입력: .hwp(OLE) 파일 경로 / 출력: extract_pdf()와 동일한 모양의 딕셔너리"""
    if not HAS_OLEFILE:
        raise RuntimeError("olefile이 설치되어 있지 않습니다 (pip install olefile)")

    ole = olefile.OleFileIO(str(path))
    dirs = ole.listdir()

    header = ole.openstream("FileHeader").read()
    # FileHeader 37번째 바이트(offset 36)의 최하위 비트가 1이면 "본문이 압축되어 있다"는 뜻
    is_compressed = (header[36] & 1) == 1

    section_nums = sorted(int(d[1][len("Section"):]) for d in dirs if d[0] == "BodyText")

    paragraphs: list[str] = []
    for num in section_nums:
        raw = ole.openstream(f"BodyText/Section{num}").read()
        # -15: zlib 헤더/체크섬 없는 "raw deflate" 모드로 풀라는 뜻 (HWP는 이 방식으로 압축함)
        data = zlib.decompress(raw, -15) if is_compressed else raw

        i, size = 0, len(data)
        while i < size:
            rec_header = struct.unpack_from("<I", data, i)[0]
            # 레코드 헤더 4바이트(32비트) = 태그(하위 10비트) + 레벨(다음 10비트) + 길이(상위 12비트)
            tag = rec_header & 0x3FF
            length = (rec_header >> 20) & 0xFFF
            # length==0xFFF(12비트 최댓값)면 다음 4바이트에 진짜 길이가 따로 있다는 뜻인데,
            # 문단 텍스트 레코드에서는 실무상 거의 안 나오는 경우라 v1에서는 미구현으로 남겨둠
            # (매우 큰 단일 문단이 있는 극소수 파일에서만 그 지점 이후 텍스트가 누락될 수 있음)
            if tag == _HWP_PARA_TEXT_TAG and length != 0xFFF:
                paragraphs.append(_decode_hwp_para_text(data[i + 4: i + 4 + length]))
            if length == 0xFFF:
                break  # 안전하게 이 섹션까지만 살리고 중단
            i += 4 + length

    text = "\n".join(p for p in paragraphs if p.strip())
    return {
        "file_path": str(path),
        "page_count": len(section_nums),
        "text": text,
        "char_count": len(text),
        "table_count": 0,  # 표 추출은 v1 미구현 (본문 텍스트만)
        "tables": [],
    }


# %%
#2-2. 신버전 .hwpx(zip + XML) 텍스트 추출
#
# [주의 - 실측으로 확인한 함정] hwpx 안에는 Preview/PrvText.txt라는 이미 UTF-8 평문인
# "미리보기 텍스트"가 있어서 이거면 충분한 줄 알았지만, 실제로는 탐색기 미리보기용으로
# 앞부분만 잘라둔 것(예: 1,022자)이라 진짜 본문(section0.xml, 1,651자)보다 짧음.
# 그래서 번거롭더라도 Contents/section*.xml을 직접 파싱해야 함.
_HWPX_NS_PARAGRAPH = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def extract_hwpx(path: Path) -> dict:
    """입력: .hwpx 파일 경로 / 출력: extract_pdf()와 동일한 모양의 딕셔너리"""
    z = zipfile.ZipFile(path)
    # 섹션이 여러 개(section0.xml, section1.xml, ...)일 수 있어 번호순 정렬 후 이어붙임
    # (data/raw 44개 정상 hwpx 중 5개가 2~4개 섹션을 가짐 - 실측 확인됨)
    section_names = sorted(
        (n for n in z.namelist() if n.startswith("Contents/section") and n.endswith(".xml")),
        key=lambda n: int("".join(c for c in n if c.isdigit()) or 0),
    )

    paragraphs: list[str] = []
    for name in section_names:
        root = ET.fromstring(z.read(name))
        # {네임스페이스}태그명 형식으로 찾아야 함 - hwpx의 <hp:t>는 실제로는 위 네임스페이스에
        # 속한 "t" 태그이기 때문
        for el in root.iter(f"{{{_HWPX_NS_PARAGRAPH}}}t"):
            if el.text:
                paragraphs.append(el.text)

    text = "\n".join(paragraphs)
    return {
        "file_path": str(path),
        "page_count": len(section_names),
        "text": text,
        "char_count": len(text),
        "table_count": 0,
        "tables": [],
    }


# %%
#3. metadata.csv 읽어서 수집된 공고 목록 불러오기
def load_metadata_rows() -> list[dict]:
    with open(METADATA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        #아까 만들어둔 list 에 목록 담기
        return list(reader)

# %%
#4. 전체 배치 처리: metadata.csv 순회하며 PDF/HWP/HWPX를 추출해서 결과를 하나의 JSON으로 저장
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
        fmt = classify_format(file_path)

        if fmt == "unsupported":
            #파일마다 print 하지 않고 확장자별 개수만 누적 (220건이면 220줄 찍혀서 안 보임)
            #(png/jpg 등 이미지는 OCR이 필요한데 이 PC에 Tesseract가 없어 v1에서는 보류함)
            skipped_unsupported_by_ext[file_path.suffix.lower()] += 1
            continue

        #포맷별로 실제 추출 함수만 다름 - 나머지 흐름(메타데이터 합치기/결과 저장)은 동일
        try:
            if fmt == "pdf":
                extracted = extract_pdf(file_path)
            elif fmt == "hwp_ole":
                extracted = extract_hwp_ole(file_path)
            else:  # "hwpx_zip"
                extracted = extract_hwpx(file_path)
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

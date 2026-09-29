# [WBS 3.3~3.6] extracted.json + metadata.csv -> 청킹 -> IDX_SUPPORT_CHUNK 스키마 레코드
#
# (이번 파일은 예외적으로 제가 직접 작성했습니다 - 원래는 확인용/에 학습용으로만 쓰고
#  src/는 직접 안 건드리는 방식으로 진행해왔는데, 이번 한 번은 하예님이 직접 요청하셔서
#  src/preprocessing/chunker.py에 바로 작성했습니다. 다음 파일부터는 다시 확인용/ 방식으로
#  돌아갑니다.)
#
# 로직은 확인용/청킹 파이프라인 뼈대.py에서 574건 실데이터로 이미 검증된 것과 동일합니다.
# 아래 주석은 "이 줄이 무슨 일을 하는지" + "실제로 어떤 값이 나오는지"를 한 줄씩 달았습니다.

import csv       # metadata.csv를 딕셔너리 형태로 읽기 위한 표준 라이브러리
import json      # extracted.json을 읽고, 최종 결과를 chunks.json으로 쓰기 위함
import re        # 문장 분리 / 기간 파싱 / 금액 추출에 정규식(regex)을 쓰기 위함
import statistics  # 마지막 리포트에서 토큰 수 중앙값(median)을 계산하기 위함
from pathlib import Path  # OS(윈도우/맥/리눅스)에 상관없이 경로를 다루기 위한 표준 라이브러리

try:
    import tiktoken
    # cl100k_base: OpenAI 계열 모델이 쓰는 토크나이저. "이 텍스트가 실제로 토큰 몇 개인지"를
    # 정확히 셀 수 있음 (한글 1글자가 토큰 1개가 아니라 보통 1.5~2개씩 소모되는 걸 반영)
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    # tiktoken이 설치 안 돼 있어도 파이프라인 전체가 죽으면 안 되므로 None으로 두고
    # count_tokens()에서 근사치 계산으로 대체 처리함
    # [주의 - 2026-09-22 WBS 8.4에서 확인] 프로젝트 .venv에는 tiktoken이 설치돼 있지 않아서, 현재 chunks.json의
    # token_count는 3,884개 전부 이 근사치(글자수/1.7)입니다. 즉 "500~800토큰" 청크는 실제로는 약 850~1,360글자입니다.
    # tiktoken을 설치하고 이 스크립트를 다시 돌리면 청크 경계가 바뀌므로, 임베딩·색인·평가도 다시 해야 합니다.
    _ENCODER = None


# ── 경로 설정 ────────────────────────────────────────────────────
# __file__ = 이 파일(chunker.py) 자신의 절대경로. resolve()로 심볼릭 링크/상대경로 정리.
THIS_FILE = Path(__file__).resolve()
# .parent = src/preprocessing/, .parent.parent = src/ 의 부모 = 프로젝트 루트
# (fetch_bizinfo.py가 cwd 기준 상대경로 때문에 겪었던 버그를 반복하지 않으려고
#  항상 "이 파일 위치" 기준으로 고정합니다)
PROJECT_ROOT = THIS_FILE.parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"                       # fetch_bizinfo.py가 다운로드한 원본 파일들
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"           # extractor.py, chunker.py의 결과물이 쌓이는 곳
EXTRACTED_JSON = PROCESSED_DIR / "extracted.json"              # extractor.py의 출력 (533건 - PDF/HWP/HWPX 텍스트+표, 그중 42건은 텍스트 없음)
METADATA_CSV = RAW_DIR / "metadata.csv"                        # fetch_bizinfo.py의 출력 (574건, 공고 메타데이터)
OUT_JSON = PROCESSED_DIR / "chunks.json"                       # 이 파일(chunker.py)의 최종 출력

# 청킹 파라미터 - 한 곳에 모아둬서 나중에 튜닝할 때 이 3줄만 보면 되게 함
CHUNK_MIN_TOKENS = 500      # 이보다 작으면 청크를 아직 안 닫고 계속 문장을 더 채움
CHUNK_MAX_TOKENS = 800      # 이걸 넘기려는 순간 청크를 닫음
CHUNK_OVERLAP_TOKENS = 100  # 청크 경계마다 앞 청크 끝부분을 다음 청크 앞에 이만큼 겹쳐 넣음


def count_tokens(text: str) -> int:
    """
    입력: 임의의 문자열 (예: "안녕하세요 지원사업 공고입니다")
    출력: 그 문자열의 토큰 개수 (정수). tiktoken 있으면 정확한 값, 없으면 글자수/1.7로 근사.
    예) "청년창업 지원사업 공고" (11글자) -> tiktoken 있으면 실제 인코딩해서 셈(보통 8~12 사이),
        없으면 int(11/1.7) = 6 으로 근사.
    """
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))  # encode()는 [12451, 8394, ...] 같은 토큰ID 리스트를 반환 -> 길이만 씀
    return int(len(text) / 1.7)


def split_sentences(text: str) -> list[str]:
    """
    입력: 문서 전체 텍스트 (extracted.json의 doc["text"], 줄바꿈이 섞인 긴 문자열)
    출력: 문장 단위로 쪼갠 문자열 리스트 (각 원소는 항상 CHUNK_MAX_TOKENS 이하가 되도록 보장)
    예) 입력 "사업개요\n지원대상: 서울 소재 소상공인. 신청기간은 8/20~9/11이다."
        -> 1) "\n" 기준으로 문단 분리: ["사업개요", "지원대상: 서울 소재 소상공인. 신청기간은 8/20~9/11이다."]
        -> 2) 각 문단을 "." "?" "!" 뒤에서 다시 분리
        -> 3) 그래도 CHUNK_MAX_TOKENS를 넘는 조각이 있으면 _split_long_segment()로 추가 분리
        -> 최종: ["사업개요", "지원대상: 서울 소재 소상공인.", "신청기간은 8/20~9/11이다."]

    [수정 2026-09-03] 원래는 2)에서 끝났는데, 확인용/정제 데이터 검증.py로 574건 실데이터를
    돌려보니 청크의 88%(2,205/2,507)가 CHUNK_MAX_TOKENS(800)를 넘는 문제가 발견됨.
    3)은 그 조사 과정에서 같이 넣은 방어 로직(마침표 없이 800토큰 넘는 문단이 실제로
    있을 수도 있으니)이지만, 디버깅해보니 574건 데이터에서 이 조건에 걸리는 문장은 0건이었음
    - 즉 3)은 "혹시 몰라서" 넣은 안전장치일 뿐, 88% 문제의 진짜 원인은 아니었음.
    진짜 원인은 chunk_sentences()에 있었음 - 거기 주석 참고.
    """
    # split("\n") -> 줄바꿈 기준으로 나눈 리스트. strip()으로 앞뒤 공백 제거, 빈 줄은 버림(if p.strip())
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        # (?<=[.?!])\s+ : "마침표/물음표/느낌표 바로 뒤에 오는 공백"을 기준으로 자름
        # lookbehind라서 구분자(.?!) 자체는 잘린 문장 끝에 그대로 남음
        parts = re.split(r"(?<=[.?!])\s+", para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.extend(_split_long_segment(part))  # 여전히 너무 길면 여기서 추가로 쪼갬
    return sentences


# 글머리기호(- · ㅇ ○ □ ▪)나 번호매김(①②③.., 1) 2) ..) 앞에서 잘라내기 위한 패턴.
# lookahead((?=...))라서 구분자 자체는 다음 조각의 맨 앞에 남고 글자는 안 버려짐 (strip()으로 다듬음)
_BULLET_PATTERN = re.compile(r"(?=(?:^|\s)[-·ㅇ○□▪]|[①②③④⑤⑥⑦⑧⑨⑩]|\d+[.)]\s)")


def _split_long_segment(segment: str) -> list[str]:
    """
    입력: split_sentences()에서 마침표로는 안 잘린 조각 하나
          (예: "ㅇ 신청자격: 서울 소재 소상공인 ㅇ 제출서류: 사업자등록증 ㅇ 접수방법: 온라인" 처럼
          표를 텍스트로 뽑아서 문장부호 없이 나열만 이어진 경우)
    출력: 각 원소가 CHUNK_MAX_TOKENS 이하인 문자열 리스트
          (원래 길이가 이미 상한 이하면 손대지 않고 [segment] 그대로 반환 - 대부분의 정상 문장은 여기서 끝남)
    동작:
      1) 상한 이하면 그대로 반환
      2) 글머리기호/번호 앞에서 분리 시도 -> 쪼갠 조각도 여전히 길 수 있어 재귀적으로 다시 검사
      3) 글머리기호도 없는 순수 긴 문단이면(최후 수단) 공백(어절) 단위로 상한을 안 넘게 강제 분할
         -> 의미 단위 보존보다 "상한을 절대 안 넘긴다"는 불변조건을 우선시함
    """
    if count_tokens(segment) <= CHUNK_MAX_TOKENS:
        return [segment]

    bullet_parts = [p.strip() for p in _BULLET_PATTERN.split(segment) if p.strip()]
    if len(bullet_parts) > 1:
        result: list[str] = []
        for part in bullet_parts:
            result.extend(_split_long_segment(part))  # 쪼갠 조각이 아직도 길 수 있어 재귀 검사
        return result

    # 글머리기호가 없어서 2)가 못 쪼갠 경우 -> 어절(공백) 단위로 강제 분할 (최후 수단)
    words = segment.split()
    forced: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = count_tokens(word)
        if current_tokens + word_tokens > CHUNK_MAX_TOKENS and current:
            forced.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        forced.append(" ".join(current))
    return forced


def chunk_sentences(sentences: list[str]) -> list[dict]:
    """
    입력: split_sentences()가 만든 문장 리스트
    출력: [{"chunk_text": "문장1 문장2 ...", "token_count": 723}, {...}, ...] 형태의 리스트
    동작: 문장을 하나씩 누적하다가 500~800토큰 사이에서 청크를 닫고, 다음 청크 시작 부분에
          직전 청크 끝의 최대 100토큰을 겹쳐서(overlap) 넣음.
    overlap을 넣는 이유: 청크 경계에서 "이 조건은 앞 문단 얘기였는데.." 하는 식으로 문맥이
    끊기는 걸 줄이기 위함 (RAG 청킹의 표준 기법).

    [수정 2026-09-03 - 88% 초과 문제의 진짜 원인] 처음엔 "문장이 너무 길어서"(-> split_sentences
    3단계 추가) 또는 "500 미만일 때 닫지 않아서"(-> MIN 조건 제거)라고 생각했는데, 둘 다 고쳐도
    수치가 1도 안 바뀌어서 (median=847, max=1008 그대로) 직접 디버깅함. 실제 원인:

        기존 코드는 문장을 하나씩 count_tokens()로 잰 값을 current_tokens에 "누적(덧셈)"만 하다가,
        닫을 때 " ".join(current)를 다시 count_tokens()로 재계산해서 chunk_text/token_count에 씀.
        문제는 tiktoken(BPE)이 "문장을 각각 따로 토큰화해서 더한 값"과 "이어붙인 뒤 한 번에
        토큰화한 값"이 다르다는 것 - 실제로 확인해보니 문장 53개를 개별로 세서 더하면 800인데,
        그 53개를 공백으로 이어붙여서 한 번에 세면 854가 나옴 (+overlap까지 겹치면 1008까지 감).
        즉 "청크를 닫을지 말지 판단하는 값(누적 합)"과 "실제로 저장되는 값(재토큰화 값)"이 서로
        다른 걸 재고 있었던 게 근본 원인 - 문장 길이나 MIN 조건은 애초에 문제가 아니었음.

    고친 방법: 매 판단마다 "닫았을 때 실제로 저장될 값"과 완전히 동일한 방식(join 후 count_tokens)
    으로 확인함. 계산량이 늘긴 하지만(문장마다 join+재토큰화) 574건 배치가 몇 초 더 걸리는 것과
    "설계 상한을 실제로 지키는 것" 중에서는 후자가 맞다고 판단함.
    """
    def joined_tokens(sents: list[str]) -> int:
        # "이 문장들을 지금 닫으면(또는 이어붙이면) 실제로 몇 토큰이 되는가"를 저장될 때와
        # 동일한 방식으로 계산 - 이게 위 버그를 고치는 핵심 (누적 합이 아니라 항상 재계산)
        return count_tokens(" ".join(sents))

    chunks: list[dict] = []
    current: list[str] = []  # 지금 채우고 있는 청크에 들어갈 문장들

    def close_chunk() -> None:
        text = " ".join(current)
        chunks.append({"chunk_text": text, "token_count": count_tokens(text)})

    for sentence in sentences:
        # "지금 문장까지 합치면 실제로 몇 토큰이 되는지"를 join+재토큰화로 직접 확인.
        # (current가 비어있는 첫 문장에서는 절대 닫지 않도록 `current` 체크로 보호)
        if current and joined_tokens(current + [sentence]) > CHUNK_MAX_TOKENS:
            close_chunk()
            # 방금 닫은 청크의 끝부분에서, "합쳤을 때 CHUNK_OVERLAP_TOKENS를 넘지 않는 선까지"
            # 뒤에서부터 문장을 하나씩 앞으로 끌어와 overlap을 만듦 (역시 join 기준으로 판단)
            overlap: list[str] = []
            for s in reversed(current):  # 뒤에서부터 훑어서 "가장 최근 문맥"이 다음 청크 앞에 오게 함
                candidate = [s] + overlap
                if joined_tokens(candidate) > CHUNK_OVERLAP_TOKENS:
                    break
                overlap = candidate
            # overlap + 다음 문장을 합쳐도 여전히 800을 넘을 수 있으므로(문장 자체가 클 경우),
            # 오래된 overlap 문장부터 빼서(pop(0)) 상한을 다시 지킴
            while overlap and joined_tokens(overlap + [sentence]) > CHUNK_MAX_TOKENS:
                overlap.pop(0)
            current = overlap
        current.append(sentence)

    if current:  # 마지막에 덜 채워진 채로 남은 문장들도 청크 하나로 마무리
        close_chunk()
    return chunks


def load_metadata_by_id() -> dict[str, dict]:
    """
    입력: 없음 (METADATA_CSV 경로를 읽음)
    출력: {"PBLN_000000000115994": {"pblancId": "...", "title": "...", "period": "...", ...}, ...}
          pblancId를 key로 하는 딕셔너리 -> 나중에 O(1)로 빠르게 조회하기 위함 (매번 574건을
          순회하지 않도록)
    """
    with open(METADATA_CSV, newline="", encoding="utf-8-sig") as f:
        # utf-8-sig: 엑셀에서 저장한 CSV 앞에 붙는 BOM(﻿) 문자를 자동으로 제거해줌
        # DictReader: 첫 줄을 헤더로 써서 각 행을 {"컬럼명": "값"} 딕셔너리로 반환
        return {row["pblancId"]: row for row in csv.DictReader(f)}


# [결정 2 - 2026-09-02] 접수기간 파싱: "YYYY-MM-DD ~ YYYY-MM-DD" 형태만 인식.
# 574건 중 284건(49%)이 "예산 소진시까지" 같은 자유텍스트라 못 쪼개는데, 이 경우 억지로
# 오늘 날짜 등을 채우지 않고 (None, None)을 돌려줌 -> 원문은 build_chunk_record()에서
# apply_period_raw에 그대로 보존 (조용히 정보를 버리지 않기 위함). 요구사항 정의서 SFR-002
# 비고란에 이 결정이 날짜와 함께 기록돼 있음.
_PERIOD_PATTERN = re.compile(
    r"(\d{4})[.\-](\d{2})[.\-](\d{2})\s*~\s*(\d{4})[.\-](\d{2})[.\-](\d{2})"
)


def parse_period(period_raw: str) -> tuple[str | None, str | None]:
    """
    입력 예1) "2026-08-20 ~ 2026-09-11"
    출력 예1) ("2026-08-20", "2026-09-11")
    입력 예2) "예산 소진시까지"
    출력 예2) (None, None)  <- 패턴이 안 맞으므로 match()가 None을 반환 -> 그대로 (None, None)
    """
    m = _PERIOD_PATTERN.match(period_raw.strip())
    if not m:
        return None, None
    y1, mo1, d1, y2, mo2, d2 = m.groups()  # groups()는 정규식 괄호 6개에 매칭된 문자열 6개를 튜플로 반환
    return f"{y1}-{mo1}-{d1}", f"{y2}-{mo2}-{d2}"


# [결정 3 - 2026-09-02] 지원금액 힌트: 정규식 매칭이라 오탐 가능성 있음 (문맥 무시하고
# 본문에서 처음 매칭되는 숫자만 잡음). 그래서 테이블정의서의 공식 컬럼명 amount 대신
# amount_hint로 저장해서, 필드명만 보고도 "확정값이 아니라 참고값"임을 알 수 있게 함.
# [숙제] 테이블정의서(IDX_SUPPORT_CHUNK)에는 아직 amount_hint 컬럼이 정의돼 있지 않으므로,
# 나중에 테이블정의서에도 이 필드를 추가하거나 amount 컬럼 설명을 보강해야 함.
_AMOUNT_PATTERN = re.compile(r"(\d[\d,]*)\s*(백만원|만원|천만원|억원)")


def extract_amount_hint(text: str) -> str | None:
    """
    입력 예1) "...최대 5,000만원(총 사업비의 70% 이내)까지 지원..."
    출력 예1) "5,000만원"  <- search()는 문자열 전체에서 첫 매칭만 찾음 (본문 뒤쪽에 다른
              금액이 더 있어도 무시됨 - 이게 위에서 말한 "오탐/누락 가능성"의 실체)
    입력 예2) "지원금액은 예산 범위 내 별도 공지"
    출력 예2) None  <- 패턴에 맞는 숫자+단위 조합이 아예 없음
    """
    m = _AMOUNT_PATTERN.search(text)
    return f"{m.group(1)}{m.group(2)}" if m else None


def build_chunk_record(*, doc: dict, meta: dict, chunk: dict, index: int) -> dict:
    """
    입력: 문서 하나(doc, extracted.json의 원소 1개), 그 문서의 metadata.csv 행(meta),
          그 문서에서 나온 청크 하나(chunk, chunk_sentences()가 만든 원소 1개), 청크 순번(index)
    출력: 테이블정의서 IDX_SUPPORT_CHUNK 스키마에 맞춘 딕셔너리 1개 (chunks.json의 원소 1개가 됨)
    예)
        {
          "chunk_id": "PBLN_000000000115994_0",
          "program_id": "PBLN_000000000115994",
          "program_name": "2026년 서울 청년창업 지원사업 공고",
          "region_name": "서울",
          "category": "창업",
          "target": "서울 소재 예비창업자",
          "apply_start": "2026-08-20", "apply_end": "2026-09-11",
          "apply_period_raw": "2026-08-20 ~ 2026-09-11",
          "amount_hint": "5,000만원",
          "chunk_text": "...", "token_count": 731,
          "embedding": None,
          "source_url": "https://www.bizinfo.go.kr/..."
        }
    """
    apply_start, apply_end = parse_period(meta["period"])
    return {
        "chunk_id": f"{doc['pblancId']}_{index}",   # 사업ID_청크순번(프로그램 단위로 이어짐) -> 같은 사업의 청크끼리 묶어 볼 때 유용
        "program_id": doc["pblancId"],
        "program_name": doc["title"],
        "region_name": doc["region"],
        "category": meta["category"],
        "target": meta["target"],
        "apply_start": apply_start,
        "apply_end": apply_end,
        "apply_period_raw": meta["period"],
        "amount_hint": extract_amount_hint(doc["text"]),
        "chunk_text": chunk["chunk_text"],
        "token_count": chunk["token_count"],
        "embedding": None,  # WBS 4단계(색인)에서 임베딩 모델 태워서 이 자리를 벡터로 채움
        "source_url": meta["source_url"],
    }


def process_all() -> None:
    """
    입력: 없음 (EXTRACTED_JSON, METADATA_CSV 파일을 읽음)
    출력: 없음 (data/processed/chunks.json 파일을 생성하고, 콘솔에 처리 요약을 출력함)
    """
    with open(EXTRACTED_JSON, encoding="utf-8") as f:
        extracted_docs = json.load(f)  # extracted.json 전체를 파이썬 리스트[딕셔너리]로 로드 (352건)
    metadata_by_id = load_metadata_by_id()  # {pblancId: 메타데이터행} 딕셔너리 (574건)

    results: list[dict] = []       # 최종적으로 chunks.json에 저장될 청크 레코드들
    skipped_empty_text = 0         # 스캔본 등 텍스트가 실질적으로 없는 문서 개수
    skipped_no_metadata = 0        # metadata.csv에 없는 pblancId 개수 (정상 흐름에서는 0이어야 함)
    # program_id(사업)별로 이어지는 청크 순번. enumerate(chunks)로 문서마다 0부터 다시 세면,
    # 같은 사업에 첨부문서가 2개 이상 있을 때 두 문서의 첫 청크가 똑같이 "..._0"이 되어
    # chunk_id가 겹칠 수 있음 - [수정 2026-09-03] 지금 574건에서는 우연히 사업당 문서가
    # 1개씩이라 안 걸렸지만(확인용/정제 데이터 검증.py로 중복 0건 확인), 구조적으로는 잠재
    # 버그였으므로 문서 경계와 상관없이 프로그램 단위로 이어지는 카운터로 바꿈
    chunk_index_by_program: dict[str, int] = {}

    for doc in extracted_docs:
        if not doc["text"].strip():
            # [수정 2026-09-03] 원래는 char_count==0만 걸렀는데, 확인용/정제 데이터 검증.py로
            # 확인해보니 char_count가 1~4처럼 0이 아니어도 text 내용이 "\n"뿐인 문서가 6건
            # 있었음 (예: PBLN_000000000125229, char_count=4, text="\n\n\n\n") - 이 6건은
            # 스킵 카운트에도 안 잡히고 조용히 청크 0개로 사라졌었음. text.strip()으로
            # "실질적인 내용이 있는지"를 보게 바꿔서, 이런 문서도 스캔본과 동일하게
            # skipped_empty_text로 정상 집계되도록 함
            skipped_empty_text += 1
            continue

        meta = metadata_by_id.get(doc["pblancId"])
        if meta is None:
            # extracted.json은 fetch_bizinfo.py가 만든 metadata.csv를 기반으로 다운로드된
            # 파일만 처리한 결과라 정상적으로는 항상 매칭돼야 하지만, 방어적으로 체크
            skipped_no_metadata += 1
            continue

        sentences = split_sentences(doc["text"])   # 문서 텍스트 -> 문장 리스트
        chunks = chunk_sentences(sentences)         # 문장 리스트 -> 500~800토큰 청크 리스트
        for chunk in chunks:
            index = chunk_index_by_program.get(doc["pblancId"], 0)
            chunk_index_by_program[doc["pblancId"]] = index + 1
            results.append(build_chunk_record(doc=doc, meta=meta, chunk=chunk, index=index))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)  # data/processed 폴더가 없으면 생성 (이미 있으면 무시)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        # ensure_ascii=False: 한글이 \uXXXX 유니코드 이스케이프로 안 바뀌고 그대로 저장되게 함
        # indent=2: 사람이 읽기 좋게 들여쓰기 (파일 용량은 커지지만 디버깅에 유리)
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 실행 리포트: 몇 건이 어떻게 처리됐는지 눈으로 확인 (감으로 넘어가지 않기 위함) ──
    token_counts = [r["token_count"] for r in results]
    print(f"[입력] extracted.json 문서 수: {len(extracted_docs)}건")
    print(f"[건너뜀] 텍스트 없음(스캔본 추정): {skipped_empty_text}건 / metadata 매칭 실패: {skipped_no_metadata}건")
    print(f"[출력] 청크 {len(results)}개 -> {OUT_JSON}")
    if token_counts:
        print(f"       토큰 수 median={statistics.median(token_counts):.0f} "
              f"min={min(token_counts)} max={max(token_counts)}")


# 이 파일을 "python chunker.py"로 직접 실행했을 때만 process_all()을 호출하고,
# 다른 파일에서 import해서 함수만 재사용할 때는 자동 실행되지 않게 막는 표준 관용구
if __name__ == "__main__":
    process_all()

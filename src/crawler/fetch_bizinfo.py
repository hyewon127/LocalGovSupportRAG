# %%
#라이브러리 선언
import os
import re
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib import response

import requests
from dotenv import load_dotenv
from pathlib import Path
from openai.lib.azure import API_KEY_SENTINEL
# %%
#0. 환경변수 : 키 값 설정
load_dotenv()
API_KEY = os.getenv("BIZINFO_API_KEY")

#기업마당 API 주소
BASE_URL = "https://www.bizinfo.go.kr/uss/rss/bizinfoApi.do"

#수집 대상 지역
TARGET_REGIONS = ["서울", "경기", "강원도", "충북", "충남", "경북", "경남", "전남광주", "전북" ,"부산","대구","세종","인천"]

#파일 ROOT 확인(parents 상위 폴더로 몇번 가는지 확인)
PROJECT_ROOT = Path.cwd().resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
# 폴더가 없을 경우 자동으로 만들어줌
RAW_DIR.mkdir(parents=True, exist_ok=True)

#API 한 번 호출 시 몇 건 씩 받을 건지
PAGE_UNIT = 100
# %%
#1. API 한 페이지 호출
def fetch_page(hashtag:str,page_index:int) -> str:
    params = {
        "crtfcKey": API_KEY,
        "dataType": "rss", #XML 로 받아서 rss
        "hashtags": hashtag,
        "pageUnit": PAGE_UNIT,
        "pageIndex": page_index,
    }
    #URL 에서 파라미터 받아오고 제한 시간은 10초
    response = requests.get(BASE_URL, params=params, timeout=10)
    #응답 상태 에러일 경우 예외 기능 발생함
    response.raise_for_status()
    # 인코딩
    response.encoding = "utf-8"
    #서버가 준 파라미터를 text로 받음
    return response.text

# %%
#2. xml 파싱
def parse_items(xml_text: str):
    # 문자열 xml 파일을 파이썬이 이해할 수 있게 트리 구조(root) 로 변환함
    root = ET.fromstring(xml_text)
    # 데이터 넣은 빈 리스트 생성
    items = []

    #channel 안에 items(공고제목) 있어 반복되는 구조
    #xml 에서 item 태그로 감싸진 공고 데이터를 모두 찾음(findall)
    for item in root.findall(".//item"):
        def get_text(tag:str) -> str:
            #내용 없을 경우 오류 방지용으로 헬퍼 함수 넣음
            el = item.find(tag)
            # 안전 장치: 태그를 찾았고 안에 글자가 있으면 text 없으면 "" 빈값
            return el.text.strip() if (el is not None and el.text) else ""

        #아이템 변수 넣기
        items.append({
            "pblancId": get_text("pblancId"),                # 공고 고유 ID
            "pblancNm": get_text("pblancNm"),                 # 공고명(사업명)
            "jrsdInsttNm": get_text("jrsdInsttNm"),           # 소관기관명(지자체명)
            "reqstBeginEndDe": get_text("reqstBeginEndDe"),   # 신청기간
            "trgetNm": get_text("trgetNm"),                   # 지원대상
            "pldirSportRealmLclasCodeNm": get_text("pldirSportRealmLclasCodeNm"),  # 지원분야
            "pblancUrl": get_text("pblancUrl"),               # 공고 상세 웹페이지 URL
            "printFlpthNm": get_text("printFlpthNm"),         # 본문 출력 파일(PDF) 경로 - 항상 존재하는 필드
            "printFileNm": get_text("printFileNm"),           # 본문 출력 파일명
        })

    # totCnt(전체 건수) 페이지 수집 종료 판단
    # API 응답에는 보통 전체 데이터가 몇개 뜻하는 totCnt 태그가 포함됨.
    # xml 파일 안에 전체 데이터가 몇개 있지 알려주는 태크를 찾아서 tot_Cnt_el 에 넣음
    tot_cnt_el = root.find(".//totCnt")
    # 숫자로 되어 있으니 int 로 추출함. 그리고 있으면 표시 없으면 0 으로 표시하게 helper 설정 !
    tot_cnt = int(tot_cnt_el.text) if (tot_cnt_el is not None and tot_cnt_el.text) else 0

    # items 이랑 tot_cnt 결과 튜플(,) 한번에 반환: 사용하는 이유는 여러 개의 데이터를 묶어서 가져오기 때문
    return items, tot_cnt

# %%
#3. 파일 다운로드
#지원사업 파일 다운로드
def download_file(url: str, save_path: Path) -> bool:
    # 파일 저장 성공하면 true, 실패 false
    if not url:
        return False
    try:
        #url 에서 가져오는 제한 시간 15초
        response = requests.get(url, timeout=15)
        #응답 상태 에러일 경우 예외 기능 발생함
        response.raise_for_status()
        #PDF 일 경우 바이너리 이기에 write_bytes 사용
        save_path.write_bytes(response.content)
        return True
    except requests.RequestException as e:
        print(f"경고: 다운로드 실패 - {url} ({e})")
        return  False
# %%
#4. 해당 지역에 대한 전체 수집 루프
def collect_region(hashtag: str, metadata_rows: list) -> None:
    #페이지를 넘기면서 공고를 수집+다운로드
    #1페이지부터 시작
    page_index = 1
    #수집한 공고 누적 개수
    collected = 0

    #4-1. 페이지 반복
    while True:
        #fetch_page 함수를 사용하여 현재 지역, 페이지 번호 전달 -> xml_text
        xml_text = fetch_page(hashtag, page_index)

        #4-2. 데이터 파싱
        # 전달된 xml_text 를 각 변수에 넣음 -> items, tot_cnt
        items, tot_cnt = parse_items(xml_text)

        #더 이상 받아올 데이터 없으면 종료
        if not items:
            break

        #4-3. 파일 다운로드 및 파일명 정제
        for item in items:
            #확장자 추출: 서버에서 확장자가 .pdf 등을 추출하고 없으면 기본값 .pdf 사용
            ext = Path(item["printFileNm"]).suffix or ".pdf"
            # 파일명에 못쓰는 특수 문자 제거
            safe_id = re.sub(r"[^\w\-]", "_", item["pblancId"]
            save_path = RAW_DIR / f"{hashtag}_{safe_id}{ext}"

            ok = download_file(item["printFlpthNm"], save_path)

            metadata_rows.append({
                "region": hashtag,
                "pblancId": item["pblancId"],
                "title": item["pblancNm"],
                "institution": item["jrsdInsttNm"],
                "period": item["reqstBeginEndDe"],
                "target": item["trgetNm"],
                "category": item["pldirSportRealmLclasCodeNm"],
                "source_url": item["pblancUrl"],
                "file_path": str(save_path) if ok else "",
                "downloaded": ok,
            })

        collected += len(items)
        print(f"[{hashtag}] {collected}/{tot_cnt} 건 수집")

        # 전체 건수를 받았거나, 마지막 페이지 개수가 pageUnit 보다 적으면 종료
        if collected >= tot_cnt or len(items) < PAGE_UNIT:
            break

        page_index += 1
        # 서버에 연속으로 요청하지 않게 쉬어가는 시간(토큰 에방)
        time.sleep(0.3)

# %%
#5. 수집 시작(100건 씩)
def main():
    all_metadata = []
    for region in TARGET_REGIONS:
        print(f"=== {region} 수집 시작 ===")
        collect_region(region, all_metadata)

    #메타데이터 CSV 로 기록
    csv_path = RAW_DIR / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"\n {len(all_metadata)}건 수집 완료 -> {csv_path}")

if __name__ == "__main__":
    main()
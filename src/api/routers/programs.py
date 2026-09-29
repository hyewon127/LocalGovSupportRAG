# [WBS 6.3] /programs 엔드포인트 - 지원사업 목록/상세 조회 (화면정의서 SCR-02 "지원사업 상세 화면"용)
#
# 분석모델 정의서 1.1의 서비스 목록 "/programs (지원사업 목록/상세)"를 구현합니다.
# /chat이 "질문 -> 답변"이라면, 여기는 LLM 없이 인덱스에 있는 사업 정보를 그대로 보여주는 조회용 API입니다.
# 그래서 LLM 키가 없어도 100% 동작합니다.
#
# [라우트 선언 순서가 중요한 이유] FastAPI는 위에서부터 차례로 경로를 맞춰봅니다.
#   /{program_id}를 /filters보다 먼저 선언하면 GET /programs/filters 요청이 program_id="filters"로
#   해석되어 "없는 사업" 404가 납니다. 그래서 고정 경로(/filters)를 변수 경로(/{program_id})보다 먼저 둡니다.

from fastapi import APIRouter, Depends, HTTPException, Query

from query_slots import TARGET_KEYWORDS_SNAPSHOT  # src/rag/query_slots.py

from ..dependencies import AppResources, get_resources
from ..program_service import get_program, list_programs
from ..schemas import FilterOptions, ProgramDetail, ProgramListResponse

router = APIRouter(prefix="/programs", tags=["programs"])


@router.get("/filters", response_model=FilterOptions)
def filter_options(res: AppResources = Depends(get_resources)):
    """
    화면정의서 SCR-01 좌측 필터 패널의 드롭다운 선택지.
    화면(Streamlit)에 "서울", "창업" 같은 값을 하드코딩하면 WBS 10(지역 확장) 때 화면 코드까지 고쳐야 하므로,
    서버 시작 시 인덱스에서 집계해둔 값(lifespan의 candidates)을 그대로 내려줍니다.
    targets는 index_mapping.py상 text 타입이라 집계가 안 돼서, query_slots.py의 사람이 추린 목록을 씁니다.
    """
    return FilterOptions(
        regions=res.candidates["region"],
        categories=res.candidates["category"],
        targets=TARGET_KEYWORDS_SNAPSHOT,
    )


@router.get("", response_model=ProgramListResponse)
def programs(
    region: str | None = None,
    category: str | None = None,
    q: str | None = Query(None, max_length=100, description="사업명 부분 검색"),
    page: int = Query(1, ge=1),
    # size 상한을 두는 이유: size=100000 같은 요청 한 번으로 인덱스 전체를 긁어가며 서버를 느리게 만드는 걸 막음.
    size: int = Query(20, ge=1, le=100),
    res: AppResources = Depends(get_resources),
):
    # /chat과 달리 없는 region/category 값을 422로 막지 않습니다. 목록 조회에서 "조건에 맞는 게 0건"은
    # 사실 그대로의 결과라 사용자를 오해하게 만들지 않기 때문입니다 (/chat은 0건이 "근거 없음"이라는
    # 다른 의미의 답변으로 바뀌어서 막았던 것).
    total, items = list_programs(
        res.os_client, region=region, category=category, q=q.strip() if q else None, page=page, size=size,
    )
    return ProgramListResponse(total=total, page=page, size=size, items=items)


@router.get("/{program_id}", response_model=ProgramDetail)
def program_detail(
    program_id: str,
    max_chunks: int = Query(3, ge=1, le=200, description="앞에서부터 내려줄 본문 청크 수"),
    res: AppResources = Depends(get_resources),
):
    detail = get_program(res.os_client, program_id, max_chunks=max_chunks)
    if detail is None:
        raise HTTPException(404, f"지원사업을 찾을 수 없습니다: {program_id}")
    return detail

# [WBS 6.1] Pydantic 스키마 정의 - API가 "무엇을 받고 무엇을 돌려주는지"의 계약서
#
# 왜 dict를 그대로 돌려주지 않고 스키마를 따로 정의하는가:
#   1) 입력 검증: 빈 질문, 500자 넘는 질문 같은 잘못된 요청을 엔드포인트 코드에 들어오기 전에 FastAPI가
#      자동으로 422 에러로 돌려보냅니다. 검증 코드를 if문으로 직접 쓰지 않아도 됩니다.
#   2) 출력 고정: pipeline.answer_question()은 내부용 키(error, invalid_citations 등)까지 담은 dict를
#      돌려주는데, response_model을 지정하면 여기 정의한 필드만 응답에 실립니다. 내부 오류 메시지 같은 값이
#      실수로 클라이언트에 새어나가는 걸 막아줍니다.
#   3) 문서 자동화: 서버를 띄우고 /docs에 들어가면 이 스키마로 만든 API 문서(Swagger UI)가 자동으로 나옵니다.
#      WBS 6.4 Streamlit 화면을 만들 때 이 문서만 보고 요청/응답 형식을 맞출 수 있습니다.
#
# [출발점] 분석모델 정의서.pptx 1.2 "서버 요구사항"에 적힌 형식을 뼈대로 삼았습니다:
#   요청: {"question": "강남구 청년 창업 지원 있어?", "region": "강남구"}
#   응답: {"answer": "...", "sources": [{"program_id": "...", "url": "..."}]}
#   여기에 화면정의서(SCR-01 필터 패널, UI-02 답변 카드)와 요구사항 SFR-008(대화 이력)에 필요한 필드를 더했습니다.

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# pipeline.py 상단 [응답 형식] 주석에 정의된 5가지 상태값. Literal로 못 박아두면, pipeline 쪽에서 오타가 난
# 상태값(예: "no_evidnece")을 돌려줬을 때 조용히 넘어가지 않고 응답 검증 단계에서 바로 에러로 드러납니다.
ChatStatus = Literal["ok", "no_evidence", "ungrounded", "llm_unavailable", "llm_error"]


# ── 1. /chat ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, examples=["서울 소상공인 창업 자금 지원사업 알려줘"])

    # 화면에서 대화를 이어가기 위한 식별자. 첫 질문에는 비워서 보내면 서버가 새로 만들어 응답에 담아주고,
    # 이후 질문부터는 그 값을 그대로 다시 보내면 됩니다 (TB_CHAT_LOG.SESSION_ID로 저장됨).
    session_id: str | None = Field(None, max_length=64)

    # ── 화면정의서 SCR-01 좌측 필터 패널(지역/업종/관심분야)에서 사용자가 직접 고른 값 ──
    # 비워두면 질문 문장에서 자동 추출한 슬롯(query_slots.py)만 씁니다.
    # 값이 있으면 자동 추출값보다 우선합니다 - 드롭다운 선택은 사용자의 "명시적인" 의도라서,
    # 문장에서 추측한 값보다 믿을 만하기 때문입니다 (pipeline.answer_question()의 region/categories 인자).
    region: str | None = Field(None, examples=["서울"])
    categories: list[str] = Field(default_factory=list, examples=[["창업"]])
    # 업종/대상(소상공인, 중소기업 등). index_mapping.py에서 target이 text 타입이라 필터가 아니라
    # BM25 검색어 힌트로만 쓰입니다 (query_slots.py [설계 결정 1]).
    targets: list[str] = Field(default_factory=list, examples=[["소상공인"]])

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        # min_length=1은 "   "(공백만)을 통과시킵니다. 공백만 있는 질문으로 검색하면 BM25는 0건,
        # kNN은 의미 없는 벡터로 아무 문서나 가져오므로, 앞뒤 공백을 지운 뒤 다시 확인합니다.
        v = v.strip()
        if not v:
            raise ValueError("질문이 비어 있습니다.")
        return v


class Slots(BaseModel):
    """실제 검색에 적용된 필터/힌트. 화면에서 "이 조건으로 찾았어요"를 보여주거나 디버깅할 때 씁니다."""
    region: str | None = None
    categories: list[str] = []
    target_keywords: list[str] = []


class Source(BaseModel):
    """
    답변 근거 1건 = 화면정의서 UI-02 "답변 결과 카드" 1장.
    citation_index는 답변 본문의 [1], [2]..와 1:1로 대응합니다 (context_builder.py [설계 결정]).
    카드에 필요한 접수기간/지원금액을 여기 같이 실어서, 화면이 카드마다 /programs/{id}를 따로 호출하지
    않아도 되게 했습니다 (답변 1번에 카드 5장이면 추가 요청 5번 -> 응답 시간 목표 5초에 불리).
    """
    citation_index: int
    program_id: str
    program_name: str
    url: str
    category: str | None = None
    apply_period: str | None = None
    amount: str | None = None
    target: str | None = None


class ChatResponse(BaseModel):
    # 대화 로그 저장에 실패해도 답변은 돌려주도록 만들었기 때문에(routers/chat.py 참고) None일 수 있습니다.
    chat_id: int | None
    session_id: str
    status: ChatStatus
    answer: str
    sources: list[Source]
    slots: Slots
    # 분석모델 정의서 2.5 평가 지표 "평균 응답시간 5초 이내"를 화면/평가에서 바로 확인할 수 있게 응답에 포함
    response_time_ms: int


class ChatLogItem(BaseModel):
    """TB_CHAT_LOG 1행 (테이블정의서 "챗봇 대화 이력" 시트)."""
    chat_id: int
    session_id: str
    question: str
    answer: str
    status: str
    cited_program_ids: list[str]
    response_time_ms: int
    created_at: datetime  # SQLite에는 UTC로 저장됨 -> 시간대 정보가 붙은 ISO 형식으로 내려줌 (chat_log.py 참고)


class ChatHistoryResponse(BaseModel):
    session_id: str
    items: list[ChatLogItem]


# ── 2. /programs ─────────────────────────────────────────────────────
class ProgramSummary(BaseModel):
    """지원사업 1건의 핵심 정보. 목록의 한 줄이자, 상세 화면(SCR-02) 상단 카드의 내용입니다."""
    program_id: str
    program_name: str
    region_name: str
    category: str | None = None
    target: str | None = None
    # 날짜로 파싱된 값(정렬/필터용)과, 화면에 그대로 보여줄 문자열(apply_period)을 둘 다 둡니다.
    # chunker.py가 "예산 소진시까지"처럼 날짜로 못 바꾼 값은 apply_start/end가 None이라서,
    # 날짜만 내려주면 화면에 "접수기간: 없음"으로 잘못 보이기 때문입니다.
    apply_start: date | None = None
    apply_end: date | None = None
    apply_period: str
    amount: str | None = None
    source_url: str


class ProgramListResponse(BaseModel):
    total: int  # 조건에 맞는 "사업" 수 (청크 수가 아님 - program_service.py 주석 참고)
    page: int
    size: int
    items: list[ProgramSummary]


class ProgramChunk(BaseModel):
    seq: int  # 원문 안에서의 순서 (chunk_id 끝의 번호)
    chunk_id: str
    text: str


class ProgramDetail(ProgramSummary):
    chunk_count: int              # 이 사업의 전체 청크 수
    chunks: list[ProgramChunk]    # 앞에서부터 max_chunks개 (전부 내려주면 큰 공고는 청크가 160개 넘음)


class FilterOptions(BaseModel):
    """SCR-01 좌측 필터 패널의 드롭다운 선택지. 화면이 값을 하드코딩하지 않고 서버에서 받아가게 합니다."""
    regions: list[str]
    categories: list[str]
    targets: list[str]


# ── 3. 공통 ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """
    서버 상태 점검용. Streamlit 화면이 첫 로딩 때 이걸 불러서 "백엔드 꺼짐 / 검색 서버 꺼짐 / LLM 키 없음"을
    구분해 안내할 수 있게 합니다 (WBS 6.5 연동 테스트에서 가장 먼저 확인하는 지점).
    """
    status: Literal["ok", "degraded"]
    opensearch: bool
    index_docs: int | None
    llm_enabled: bool
    llm_provider: str
    llm_model: str

# [WBS 6.4] Streamlit 데모 UI - 화면정의서 SCR-01(메인 채팅) + SCR-02(지원사업 상세)
#
# 실행 방법 (백엔드를 먼저 띄운 뒤, 프로젝트 루트에서):
#   .venv\Scripts\python.exe -m streamlit run ui/app.py
#   -> 브라우저가 http://localhost:8501 로 열립니다.
#
# [화면정의서와의 대응]
#   SCR-01 메인 채팅 화면
#     ① 상단 헤더          -> render_header()   : 서비스명, 서비스 범위(서울), 공고 수
#     ② 좌측 필터 패널      -> render_sidebar()  : 지역 / 업종·대상 / 관심분야 + 서비스 상태
#     ③ 대화창 + 답변 카드  -> render_chat_view() / render_source_cards() (UI-02)
#     ④ 입력창 + 추천 질문  -> st.chat_input + EXAMPLE_QUESTIONS
#   SCR-02 지원사업 상세 화면 -> render_detail_view() : ◀ 돌아가기, 핵심 정보 카드, 공고 본문, 원문 링크
#   SCR-03 데이터 현황(관리자)은 요구사항 UI-04에서 "선택 구현"이라 이번 범위에서 제외했습니다.
#
# [Streamlit 동작 방식 - 이 파일을 읽을 때 꼭 알아야 할 것]
#   Streamlit은 버튼 클릭/입력 같은 상호작용이 생길 때마다 이 파일을 "위에서부터 끝까지 다시 실행"합니다.
#   그래서 일반 변수는 매번 초기화되고, 대화 내역처럼 유지돼야 하는 값은 전부 st.session_state에 둬야 합니다.
#   화면 전환(채팅 <-> 상세)도 페이지를 따로 만들지 않고 session_state["view"] 값으로 어느 화면을 그릴지 고릅니다
#   - 상세 화면에서 "돌아가기"를 눌렀을 때 대화 내역이 그대로 남아 있어야 해서(SCR-02 ①), 같은 세션 안에서
#   값만 바꾸는 방식이 가장 단순합니다.

import api_client  # streamlit run은 실행한 파일의 폴더(ui/)를 import 경로에 넣어주므로 바로 import 가능
import streamlit as st
from api_client import ApiError

st.set_page_config(page_title="지자체 지원사업 RAG 챗봇", page_icon="🏛️", layout="wide")

# 화면정의서 SCR-01 ④ "최근 질문 추천 표시". 실제 사용자 로그가 쌓이기 전이라 평가용으로 써본 질문 중
# 인덱스에 근거가 확실히 있는 것들을 골랐습니다 (WBS 5에서 관련 질의로 확인한 문장들).
EXAMPLE_QUESTIONS = [
    "서울 소상공인 창업 자금 지원사업 알려줘",
    "여성기업 대상 수출 지원사업 있어?",
    "인력 채용하면 받을 수 있는 지원금 알려줘",
    "AI 기술개발 지원사업 찾아줘",
]

# pipeline.py의 5가지 status별 안내. 백엔드가 status를 따로 주기 때문에(분석모델 정의서 형식에 추가한 필드)
# 화면이 "정상 답변"과 "근거 없음/LLM 사용 불가"를 구분해서 다르게 보여줄 수 있습니다.
STATUS_NOTICE = {
    "no_evidence": ("info", "관련 공고를 찾지 못했어요. 왼쪽 필터를 풀거나 질문을 조금 바꿔서 다시 물어보세요."),
    "ungrounded": ("warning", "근거를 확인할 수 없는 답변이라 표시하지 않았어요. 질문을 더 구체적으로 바꿔보세요."),
    "llm_unavailable": ("warning", "LLM API 키가 설정되지 않아, 답변 생성 없이 질문과 관련된 공고만 보여드려요."),
    "llm_error": ("error", "답변 생성 중 오류가 발생해, 질문과 관련된 공고만 보여드려요."),
}

# 마크다운에서 특별한 의미를 갖는 문자들. 공고명/본문은 PDF에서 뽑은 원문이라 "*", "_", "$", "|" 같은 문자가
# 섞여 있는데, 그대로 st.markdown에 넣으면 굵게/기울임/수식/표로 잘못 렌더링됩니다
# (예: "$"가 두 개 있으면 그 사이가 LaTeX 수식으로 바뀜). 앞에 \를 붙여 글자 그대로 보이게 합니다.
_MD_SPECIAL_CHARS = set("\\`*_{}[]()#+-.!|~<>$")


def md_escape(text: str | None) -> str:
    return "".join(f"\\{ch}" if ch in _MD_SPECIAL_CHARS else ch for ch in (text or ""))


# ── 1. 세션 상태 ─────────────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "messages": [],       # [{"role": "user", "content"} | {"role": "assistant", "result"/"error"/"restored"}]
        # 새로고침해도 대화를 이어갈 수 있게 session_id를 URL(?session=...)에도 보관합니다 (SFR-008 재조회).
        # session_state는 브라우저 새로고침 시 사라지지만 URL은 남기 때문입니다.
        "session_id": st.query_params.get("session"),
        "history_loaded": False,
        "view": "chat",       # "chat"(SCR-01) | "detail"(SCR-02)
        "program_id": None,   # 상세 화면에서 보여줄 사업
        "detail_chunks": 3,   # 상세 화면 본문을 앞에서부터 몇 구간 보여줄지
        "pending_question": None,  # 추천 질문 버튼으로 들어온 질문 (다음 실행에서 처리)
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ── 2. 백엔드 호출 (캐시) ────────────────────────────────────────────
# st.cache_data: 같은 인자로 다시 부르면 백엔드에 요청하지 않고 저장된 결과를 돌려줍니다.
# Streamlit은 상호작용마다 파일 전체를 다시 실행하므로, 캐시가 없으면 버튼 하나 누를 때마다 /health,
# /programs/filters를 매번 다시 호출합니다. 예외가 난 호출은 캐시되지 않아서, 백엔드가 꺼져 있다가 켜지면
# 다음 실행에서 바로 다시 시도됩니다.
@st.cache_data(ttl=10, show_spinner=False)
def cached_health() -> dict:
    return api_client.health()


@st.cache_data(ttl=300, show_spinner=False)
def cached_filters() -> dict:
    return api_client.get_filters()


@st.cache_data(ttl=300, show_spinner=False)
def cached_program_count() -> int:
    return api_client.count_programs()


@st.cache_data(ttl=300, show_spinner=False)
def cached_program(program_id: str, max_chunks: int) -> dict:
    return api_client.get_program(program_id, max_chunks)


# ── 3. 화면 전환 콜백 ────────────────────────────────────────────────
# on_click 콜백은 "다음 실행이 시작되기 전에" 실행됩니다. 그래서 값을 바꾸고 나면 그 다음 실행에서
# 바로 바뀐 화면이 그려집니다 (버튼 아래에서 if st.button(...)으로 처리하면 한 번 늦게 반영되는 경우가 있음).
def open_detail(program_id: str) -> None:
    st.session_state.view = "detail"
    st.session_state.program_id = program_id
    st.session_state.detail_chunks = 3


def back_to_chat() -> None:
    st.session_state.view = "chat"


def show_more_chunks() -> None:
    st.session_state.detail_chunks += 5


def ask_example(question: str) -> None:
    st.session_state.pending_question = question


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.history_loaded = True  # 방금 비운 대화를 URL의 옛 session으로 다시 불러오지 않도록
    st.query_params.clear()


# ── 4. 공통 영역: 헤더 / 좌측 패널 ───────────────────────────────────
def render_header(health: dict) -> None:
    st.title("🏛️ 지자체 지원사업 RAG 챗봇")
    try:
        program_count = f"{cached_program_count():,}건"
    except ApiError:
        program_count = "-건"
    st.caption(f"1차 서비스 지역: 서울특별시 · 기업마당 공고 {program_count} 기반 · 답변마다 근거 공고를 [번호]로 인용합니다")
    if health["status"] != "ok":
        st.warning("검색 서버(OpenSearch)에 연결되지 않아 질문에 답할 수 없어요. OpenSearch를 먼저 실행해 주세요.")


def render_sidebar(health: dict) -> dict:
    """좌측 필터 패널 (SCR-01 ②). 반환값은 /chat 요청에 그대로 넣을 필터 값."""
    try:
        filters = cached_filters()
    except ApiError:
        filters = {"regions": [], "categories": [], "targets": []}

    with st.sidebar:
        st.header("검색 필터")
        st.caption("비워두면 질문 문장에서 조건을 자동으로 찾아요. 직접 고르면 그 값이 우선합니다.")
        # "전체"를 첫 선택지로 두고 None으로 바꿔 보내는 이유: 백엔드는 값이 없을 때만 문장에서 추출한 슬롯을
        # 쓰므로(pipeline.answer_question), "선택 안 함"을 명확히 None으로 전달해야 합니다.
        region = st.selectbox("지역", ["전체"] + filters["regions"], key="filter_region")
        target = st.selectbox("업종·대상", ["전체"] + filters["targets"], key="filter_target")
        categories = st.multiselect("관심분야", filters["categories"], key="filter_categories",
                                    placeholder="분야 선택 (복수 가능)")

        st.divider()
        st.button("🆕 새 대화 시작", on_click=reset_conversation, width="stretch")

        st.divider()
        st.subheader("서비스 상태")
        st.markdown("- 백엔드: ✅ 연결됨")
        if health["opensearch"]:
            st.markdown(f"- 검색 서버: ✅ 정상 (청크 {health['index_docs']:,}개)")
        else:
            st.markdown("- 검색 서버: ❌ 연결 안 됨")
        if health["llm_enabled"]:
            st.markdown(f"- 답변 생성: ✅ {health['llm_provider']} / {health['llm_model']}")
        else:
            st.markdown(f"- 답변 생성: ⚠️ API 키 미설정 ({health['llm_provider']}) - 관련 공고 목록만 안내")

    return {
        "region": None if region == "전체" else region,
        "targets": [] if target == "전체" else [target],
        "categories": categories,
    }


# ── 5. SCR-01 메인 채팅 화면 ─────────────────────────────────────────
def restore_history_once() -> None:
    """
    URL에 session이 있는데 화면에 대화가 없으면(= 새로고침한 경우) 백엔드의 대화 이력(TB_CHAT_LOG)으로 복원합니다.
    [한계] TB_CHAT_LOG에는 답변 문장과 인용된 program_id만 저장되고 카드에 필요한 사업명/접수기간은 없어서,
    복원된 대화에는 출처 카드를 다시 그리지 않습니다 (카드를 살리려면 사업마다 /programs 조회가 추가로 필요).
    """
    if st.session_state.history_loaded:
        return
    st.session_state.history_loaded = True
    if not st.session_state.session_id or st.session_state.messages:
        return
    try:
        items = api_client.get_history(st.session_state.session_id)
    except ApiError:
        return
    for item in items:
        st.session_state.messages.append({"role": "user", "content": item["question"]})
        st.session_state.messages.append({"role": "assistant", "restored": item})


def group_sources(sources: list[dict]) -> list[dict]:
    """
    같은 사업이 근거에 2번 들어오는 경우(hybrid_search.py가 사업당 청크를 최대 2개까지 허용)를 카드 1장으로 합칩니다.
    카드가 똑같이 2장 나오면 사용자는 "같은 게 왜 두 번?"이라고 느끼기 때문에, 카드는 사업 단위로 묶고
    인용 번호만 [1][2]처럼 모아서 답변 본문의 번호와 계속 대응되게 합니다.
    """
    grouped: dict[str, dict] = {}
    for source in sources:
        entry = grouped.setdefault(source["program_id"], {"source": source, "citations": []})
        entry["citations"].append(source["citation_index"])
    return list(grouped.values())  # dict는 넣은 순서를 유지하므로 점수 순서가 그대로 보존됨


def render_source_cards(sources: list[dict], message_index: int) -> None:
    """UI-02 답변 결과 카드: 지원사업명 / 한줄 요약 / 접수기간 / 지원금액 / 출처 링크 + 상세 보기."""
    groups = group_sources(sources)
    for i, group in enumerate(groups):
        # 카드 2장마다 새 행(st.columns)을 만듭니다. 열 2개를 한 번만 만들고 좌/우에 번갈아 넣으면
        # 카드마다 높이가 달라서(사업명 길이 차이) 행이 어긋나고, 읽는 순서도 [1]->[3]->[2]처럼 뒤섞여 보입니다.
        if i % 2 == 0:
            row = st.columns(2)
        source = group["source"]
        citations = "".join(f"[{n}]" for n in group["citations"])
        with row[i % 2], st.container(border=True):
            st.markdown(f"**{citations} {md_escape(source['program_name'])}**")
            # "한줄 요약" 자리: 요약문 필드가 인덱스에 없어서(LLM 요약은 키가 필요) 분야·지원대상으로 대신합니다.
            summary = " · ".join(v for v in (source.get("category"), source.get("target")) if v)
            st.caption(md_escape(summary) or "분야/대상 정보 없음")
            st.markdown(
                f"📅 접수기간: {md_escape(source.get('apply_period') or '명시 없음')}  \n"
                f"💰 지원금액: {md_escape(source.get('amount') or '명시 없음')}"
            )
            left, right = st.columns(2)
            # key에 message_index를 넣는 이유: 같은 사업이 여러 답변에 나오면 버튼 key가 겹쳐서
            # Streamlit이 DuplicateWidgetID 에러를 냅니다. "몇 번째 답변의 어떤 사업"으로 유일하게 만듭니다.
            left.button("상세 보기", key=f"detail_{message_index}_{source['program_id']}",
                        on_click=open_detail, args=(source["program_id"],), width="stretch")
            right.link_button("원문 공고 ↗", source["url"], width="stretch")


def render_assistant_message(message: dict, index: int) -> None:
    if "error" in message:
        st.error(message["error"])
        return

    if "restored" in message:
        item = message["restored"]
        # 정상 답변은 실시간 답변과 똑같이 마크다운 그대로, 나머지(근거 없음 등 안내 문구)는 안내 상자로
        if item["status"] == "ok":
            st.markdown(item["answer"])
        else:
            st.info(item["answer"])
        st.caption(f"🕘 이전 대화 기록 · 인용 공고 {len(item['cited_program_ids'])}건 (출처 카드는 새 질문부터 표시)")
        return

    result = message["result"]
    notice = STATUS_NOTICE.get(result["status"])
    if notice is None:
        # 정상 답변: LLM 답변 본문 그대로. [번호] 인용이 아래 카드의 [번호]와 대응됩니다.
        # 여기는 md_escape를 하지 않습니다 - LLM이 목록/굵게 같은 마크다운으로 답을 정리해 오는 게 의도된 형식이라서.
        st.markdown(result["answer"])
    else:
        level, text = notice
        getattr(st, level)(text)

    slots = result["slots"]
    conditions = " · ".join(filter(None, [
        f"지역 {slots['region']}" if slots["region"] else None,
        f"분야 {', '.join(slots['categories'])}" if slots["categories"] else None,
        f"대상 {', '.join(slots['target_keywords'])}" if slots["target_keywords"] else None,
    ])) or "조건 없음(전체 검색)"
    # 검색 조건을 보여주는 이유: 결과가 이상할 때 "필터가 잘못 잡혀서"인지 사용자가 바로 알 수 있게 (투명성)
    st.caption(f"🔎 검색 조건: {md_escape(conditions)} · ⏱ {result['response_time_ms'] / 1000:.2f}초")

    if result["sources"]:
        render_source_cards(result["sources"], index)


def handle_question(question: str, filter_values: dict) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md_escape(question))
    with st.chat_message("assistant"), st.spinner("관련 공고를 찾고 답변을 만드는 중..."):
        try:
            result = api_client.chat(question, session_id=st.session_state.session_id, **filter_values)
        except ApiError as e:
            st.session_state.messages.append({"role": "assistant", "error": str(e)})
        else:
            st.session_state.session_id = result["session_id"]
            st.query_params["session"] = result["session_id"]
            st.session_state.messages.append({"role": "assistant", "result": result})
    # 답변을 session_state에 넣은 뒤 처음부터 다시 그립니다. 방금 받은 답변을 여기서 따로 그리면
    # "대화 내역 그리기"와 "새 답변 그리기" 코드가 두 벌이 되고, 카드 버튼 key도 두 번 만들어져 충돌합니다.
    st.rerun()


def render_chat_view(filter_values: dict, health: dict) -> None:
    restore_history_once()

    if not st.session_state.messages:
        st.markdown("#### 무엇을 찾고 계세요?")
        st.caption("예시 질문을 눌러보거나, 아래 입력창에 직접 질문해 보세요.")
        columns = st.columns(len(EXAMPLE_QUESTIONS))
        for column, question in zip(columns, EXAMPLE_QUESTIONS):
            column.button(question, on_click=ask_example, args=(question,), width="stretch")

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(md_escape(message["content"]))
            else:
                render_assistant_message(message, i)

    # OpenSearch가 꺼져 있으면 어차피 503이 나므로 입력창을 비활성화해서 헛된 요청을 막습니다.
    question = st.chat_input("질문을 입력하세요...", disabled=not health["opensearch"])
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    if question or pending:
        handle_question(question or pending, filter_values)


# ── 6. SCR-02 지원사업 상세 화면 ─────────────────────────────────────
def render_detail_view() -> None:
    st.button("◀ 대화로 돌아가기", on_click=back_to_chat)
    st.caption("홈 > 답변 > 상세")

    try:
        program = cached_program(st.session_state.program_id, st.session_state.detail_chunks)
    except ApiError as e:
        st.error(e)
        return

    st.header(md_escape(program["program_name"]))

    # SCR-02 ②: 핵심 정보(지자체/분야/접수기간/지원대상/지원금액)를 카드 형태로
    info = [
        ("지자체", program["region_name"]),
        ("분야", program["category"]),
        ("접수기간", program["apply_period"]),
        ("지원대상", program["target"]),
        ("지원금액", program["amount"]),
    ]
    for column, (label, value) in zip(st.columns(len(info)), info):
        with column, st.container(border=True):
            st.caption(label)
            st.markdown(f"**{md_escape(value or '명시 없음')}**")

    # SCR-02 ③④: 원문 링크. 인덱스의 source_url은 PDF 파일이 아니라 기업마당 공고 페이지 주소라서
    # (첨부파일 URL은 색인하지 않음), "PDF 다운로드" 대신 첨부파일이 있는 공고 페이지로 안내합니다.
    st.link_button("📄 기업마당 원문 공고 · 첨부파일 보기 ↗", program["source_url"], type="primary")

    st.subheader("공고 본문")
    st.caption(f"전체 {program['chunk_count']}개 구간 중 앞 {len(program['chunks'])}개 · "
               "PDF/HWP에서 자동 추출한 텍스트라 표나 서식이 깨져 보일 수 있어요.")
    for chunk in program["chunks"]:
        with st.container(border=True):
            # 줄바꿈 보존: 마크다운은 줄바꿈 하나를 무시하므로, 줄 끝에 공백 2칸을 붙여 강제 줄바꿈으로 바꿉니다.
            st.markdown(md_escape(chunk["text"]).replace("\n", "  \n"))
    if len(program["chunks"]) < program["chunk_count"]:
        st.button("본문 더 보기 (+5)", on_click=show_more_chunks)


# ── 7. 진입점 ────────────────────────────────────────────────────────
def main() -> None:
    init_state()

    # 백엔드가 꺼져 있으면 아무것도 할 수 없으므로, 실행 방법을 안내하고 여기서 멈춥니다.
    try:
        health = cached_health()
    except ApiError as e:
        st.title("🏛️ 지자체 지원사업 RAG 챗봇")
        st.error(str(e))
        st.stop()

    filter_values = render_sidebar(health)
    if st.session_state.view == "detail" and st.session_state.program_id:
        render_detail_view()
    else:
        render_header(health)
        render_chat_view(filter_values, health)


main()

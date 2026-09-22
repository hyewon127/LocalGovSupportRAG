# [WBS 5.3] 검색 결과 top-k 조립 및 컨텍스트 포맷팅
#
# 배경: hybrid_search()가 돌려주는 top-5 청크는 LLM이 그대로 이해하기 좋은 형태가 아닙니다
# (딕셔너리 리스트일 뿐, "이게 몇 번째 근거고 어디서 왔는지"가 텍스트로 드러나 있지 않습니다).
# 이 파일은 두 가지를 만듭니다:
#   1) context_text - LLM 프롬프트에 그대로 붙여넣을 문자열. 청크마다 [1], [2].. 번호를 매겨서,
#      WBS 5.4(출처 인용 강제 프롬프트)에서 "답변에 [번호]를 인용해라"라고 시킬 수 있게 합니다.
#   2) sources - API 응답에 그대로 실을 리스트. 분석모델 정의서.pptx(1.2 서버 요구사항)에
#      명시된 리턴 값 포맷 {"answer":"...", "sources":[{"program_id":"...","url":"..."}]}을
#      그대로 따릅니다.
#
# [설계 결정] "인용 번호(1, 2, 3..)"와 "실제 근거 리스트(sources)"의 인덱스를 반드시 1:1로 맞춥니다.
#   LLM이 답변에서 "[2]에 따르면.."이라고 썼을 때, 클라이언트가 sources[1](0-indexed로 2번째)을
#   그대로 찾아서 보여줄 수 있어야 하기 때문입니다. 번호가 어긋나면 "출처 인용"이라는 이 프로젝트의
#   핵심 요구사항(분석모델 정의서 2.5 평가 설계: 출처 인용률 95% 이상)이 실제로는 틀린 링크를
#   보여주는 결과로 이어집니다 - 그래서 build_context() 하나에서 둘을 같이 만들어 어긋날 여지를 없앴습니다.

def format_period(chunk: dict) -> str:
    """
    [WBS 8.4] 이름 앞의 _를 뗐습니다(공개 함수). src/api/program_service.py에 똑같은 함수가 한 벌 더 있었는데,
    한쪽만 고치면 챗봇 답변 카드와 지원사업 상세 화면의 접수기간 표기가 서로 달라지므로 여기 하나로 합쳤습니다.

    입력: hybrid_search()가 반환한 청크 1개 (apply_start/apply_end/apply_period_raw 포함)
    출력: 화면에 보여줄 접수기간 문자열
    chunker.py의 parse_period()가 "예산 소진시까지"처럼 날짜로 못 바꾼 값은 apply_start/end를
    None으로 두고 apply_period_raw에 원문을 보존했으므로(chunker.py 210번째 줄 주석 참고),
    여기서도 그 원문을 그대로 보여주는 걸 우선합니다(날짜가 있으면 날짜를, 없으면 원문을).
    """
    if chunk.get("apply_start") and chunk.get("apply_end"):
        return f"{chunk['apply_start']} ~ {chunk['apply_end']}"
    return chunk.get("apply_period_raw") or "명시 없음"


def build_context(chunks: list[dict]) -> dict:
    """
    입력: hybrid_search()의 반환값 (score 내림차순으로 이미 정렬된 top-k 청크 리스트)
    출력: {
        "has_results": bool,        # 5.6(가드레일)이 "근거 없음" 분기를 타야 하는지 바로 판단할 수 있게
        "context_text": str,        # LLM 프롬프트에 붙일 문자열 ([1]..[k] 번호 포함)
        "sources": list[dict],      # API 응답용, context_text의 번호와 1:1 대응 (0-indexed 리스트지만
                                     # sources[i]["citation_index"]에 실제 표시 번호(i+1)를 같이 저장)
    }
    """
    if not chunks:
        # 빈 리스트를 그냥 넘기면 아래 for문이 아무것도 안 하고 조용히 빈 문자열을 반환해버려서,
        # "검색이 안 된 건지 원래 결과가 없는 건지" 호출하는 쪽에서 구분이 안 됩니다.
        # has_results로 명시적으로 알려줘야 5.6에서 "관련 지원사업을 찾지 못했습니다" 같은
        # 정해진 응답으로 분기할 수 있습니다.
        return {"has_results": False, "context_text": "", "sources": []}

    context_blocks = []
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{i}] {chunk['program_name']} ({chunk['region_name']} / {chunk['category']})\n"
            f"- 지원대상: {chunk.get('target') or '명시 없음'}\n"
            f"- 접수기간: {format_period(chunk)}\n"
            f"- 지원금액: {chunk.get('amount_hint') or '명시 없음'}\n"
            f"- 본문: {chunk['chunk_text']}"
        )
        sources.append({
            "citation_index": i,
            "program_id": chunk["program_id"],
            "program_name": chunk["program_name"],
            "url": chunk["source_url"],
            # [WBS 6.2 추가] 화면정의서 UI-02 "답변 결과 카드"(지원사업명/접수기간/지원금액/출처 링크)를
            # 그리는 데 필요한 값. 청크에 이미 들어 있는 값이라 추가 조회 비용이 없고, 여기서 같이 실어두면
            # 화면이 카드마다 /programs/{id}를 따로 부르지 않아도 됩니다 (api/schemas.py의 Source 참고).
            "category": chunk.get("category"),
            "apply_period": format_period(chunk),
            "amount": chunk.get("amount_hint"),
            "target": chunk.get("target"),
        })

    return {
        "has_results": True,
        # 청크 사이를 빈 줄 2개로 구분: LLM이 "여기부터 새 근거 문서"라는 걸 문단 단위로 구분하기 쉽게
        "context_text": "\n\n".join(context_blocks),
        "sources": sources,
    }


# ── 수동 확인용 실행 블록 ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    _THIS_FILE = Path(__file__).resolve()
    _INDEXING_DIR = _THIS_FILE.parent.parent / "indexing"
    if str(_INDEXING_DIR) not in sys.path:
        sys.path.insert(0, str(_INDEXING_DIR))

    from config import get_client
    from hybrid_search import hybrid_search
    from query_slots import extract_slots, fetch_candidate_values

    client = get_client()
    candidates = fetch_candidate_values(client)

    query = "서울 소상공인 창업 자금 지원사업 알려줘"
    slots = extract_slots(query, candidates=candidates)
    chunks = hybrid_search(
        query, region=slots["region"], categories=slots["categories"],
        target_keywords=slots["target_keywords"], client=client,
    )
    context = build_context(chunks)

    print(f"[has_results] {context['has_results']}\n")
    print("[context_text]")
    print(context["context_text"])
    print("\n[sources]")
    for s in context["sources"]:
        print(f"  {s}")

    print("\n[빈 결과 케이스 확인]")
    empty_context = build_context([])
    print(f"  {empty_context}")

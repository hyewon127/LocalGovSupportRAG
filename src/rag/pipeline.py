# [WBS 5 통합] 질문 1개 -> 답변 1개까지 5.1~5.6을 순서대로 묶는 진입점
#
# 분석모델 정의서 1.1 시스템 구조도의 흐름을 그대로 코드로 옮겼습니다:
#   "질의를 슬롯 추출 -> OpenSearch 하이브리드 검색으로 근거 문서 조회 -> LLM에 컨텍스트를 전달해
#    출처 인용 답변 생성 -> 클라이언트에 반환"
# WBS 6.2의 FastAPI /chat 엔드포인트는 이 파일의 answer_question()만 부르면 되도록 만들었습니다.
#
# [응답 형식] 분석모델 정의서 1.2의 리턴 값 {"answer": "...", "sources": [{"program_id", "url"}]}에
#   status를 추가했습니다. status가 있어야 화면(WBS 6.4)과 평가(WBS 7.3)가 "정상 답변"과
#   "근거 없음/인용 위반/LLM 사용 불가"를 구분할 수 있습니다.
#     ok              : 정상. sources에는 답변이 실제로 인용한 근거만 있음
#     no_evidence     : 관련 공고가 없음 (검색 단계에서 걸렀거나, 모델이 스스로 근거 없음이라고 답함)
#     ungrounded      : 모델이 인용 규칙을 어겨서 답변을 버림
#     llm_unavailable : API 키가 없어 답변 생성은 못 하고, 검색된 공고만 돌려줌
#     llm_error       : LLM 호출 중 오류 (네트워크/키/모델명 등). 검색된 공고만 돌려줌

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_INDEXING_DIR = _THIS_FILE.parent.parent / "indexing"
if str(_INDEXING_DIR) not in sys.path:
    sys.path.insert(0, str(_INDEXING_DIR))

from config import get_client  # noqa: E402
from context_builder import build_context  # noqa: E402
from generator import generate_answer  # noqa: E402
from guardrail import has_enough_evidence, validate_answer  # noqa: E402
from hybrid_search import hybrid_search  # noqa: E402
from llm_client import get_llm_client  # noqa: E402
from prompt_template import NO_EVIDENCE_MESSAGE  # noqa: E402
from query_slots import extract_slots, fetch_candidate_values  # noqa: E402

LLM_UNAVAILABLE_MESSAGE = "현재 답변 생성 기능을 사용할 수 없어, 질문과 관련된 공고 목록만 안내합니다."


def answer_question(question: str, *, os_client=None, llm_client=None, candidates: dict | None = None) -> dict:
    """
    입력: question - 사용자 질문
          os_client - OpenSearch 클라이언트 (없으면 새로 만듦)
          llm_client - LLM 클라이언트 (없으면 llm_client.get_llm_client()로 만듦, 키가 없으면 None)
          candidates - query_slots.fetch_candidate_values() 결과. 없으면 매 질문마다 집계 쿼리를 한 번 더
                       보내게 되므로, FastAPI에서는 서버 시작 시 한 번만 구해서 계속 넘기는 걸 권장
    출력: {"status", "answer", "sources", "slots", ...} (상단 [응답 형식] 참고)
    """
    os_client = os_client or get_client()
    llm = llm_client if llm_client is not None else get_llm_client()
    candidates = candidates or fetch_candidate_values(os_client)

    # 5.1 슬롯 추출
    slots = extract_slots(question, candidates=candidates, llm_client=llm)

    # 5.2 하이브리드 검색
    chunks = hybrid_search(
        question,
        region=slots["region"],
        categories=slots["categories"],
        target_keywords=slots["target_keywords"],
        client=os_client,
    )

    # 5.6 [호출 전] 가드레일 - 관련 있는 근거가 없으면 LLM을 부르지 않음
    if not has_enough_evidence(chunks):
        return {"status": "no_evidence", "answer": NO_EVIDENCE_MESSAGE, "sources": [], "slots": slots}

    # 5.3 컨텍스트 조립
    context = build_context(chunks)

    if llm is None:
        # LLM이 없어도 검색은 됐으므로, 빈손으로 돌려보내지 않고 찾은 공고 목록은 보여줍니다.
        return {"status": "llm_unavailable", "answer": LLM_UNAVAILABLE_MESSAGE,
                "sources": context["sources"], "slots": slots}

    # 5.4 + 5.5 프롬프트 구성 및 답변 생성
    try:
        raw_answer = generate_answer(question, context["context_text"], llm)
    except Exception as e:
        return {"status": "llm_error", "answer": LLM_UNAVAILABLE_MESSAGE,
                "sources": context["sources"], "slots": slots, "error": str(e)}

    # 5.6 [호출 후] 가드레일 - 인용 검증
    checked = validate_answer(raw_answer, context["sources"])
    return {**checked, "slots": slots}


if __name__ == "__main__":
    client = get_client()
    candidates = fetch_candidate_values(client)
    for q in ["서울 소상공인 창업 자금 지원사업 알려줘", "오늘 날씨 어때"]:
        result = answer_question(q, os_client=client, candidates=candidates)
        print(f"\n[질문] {q}")
        print(f"  status : {result['status']}")
        print(f"  answer : {result['answer']}")
        print(f"  sources: {[s['program_name'][:30] for s in result['sources']]}")

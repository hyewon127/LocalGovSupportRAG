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


def answer_question(
    question: str,
    *,
    os_client=None,
    llm_client=None,
    candidates: dict | None = None,
    region: str | None = None,
    categories: list[str] | None = None,
    target_keywords: list[str] | None = None,
    return_trace: bool = False,
) -> dict:
    """
    입력: question - 사용자 질문
          os_client - OpenSearch 클라이언트 (없으면 새로 만듦)
          llm_client - LLM 클라이언트 (없으면 llm_client.get_llm_client()로 만듦, 키가 없으면 None)
          candidates - query_slots.fetch_candidate_values() 결과. 없으면 매 질문마다 집계 쿼리를 한 번 더
                       보내게 되므로, FastAPI에서는 서버 시작 시 한 번만 구해서 계속 넘기는 걸 권장
          region/categories/target_keywords - [WBS 6.2 추가] 화면 필터 패널(화면정의서 SCR-01)에서 사용자가
                       직접 고른 값. 비워두면 기존처럼 질문 문장에서 추출한 슬롯만 씁니다.
          return_trace - [WBS 7.3 추가] True면 결과에 "trace"(검색된 청크, 가드레일 적용 전 LLM 원본 답변)를
                       붙입니다. 평가(src/evaluation/answer_eval.py)가 "모델이 원래 뭐라고 답했는지"를 봐야
                       출처 인용률을 잴 수 있는데, 최종 answer는 가드레일이 이미 고친/버린 값이라서입니다.
                       API(schemas.ChatResponse)에는 trace 필드가 없어서 켜도 응답에 실리지 않습니다.
    출력: {"status", "answer", "sources", "slots", ...} (상단 [응답 형식] 참고)
    """
    # 평가용 중간값. 단계마다 채워 두고, 어느 단계에서 끝나든 finish()가 한 곳에서 붙입니다
    # (return 문마다 trace 코드를 넣으면 한 군데를 빠뜨리기 쉬움).
    trace = {"chunks": None, "raw_answer": None}

    def finish(result: dict) -> dict:
        if return_trace:
            result["trace"] = trace
        return result

    os_client = os_client or get_client()
    llm = llm_client if llm_client is not None else get_llm_client()
    candidates = candidates or fetch_candidate_values(os_client)

    # 5.1 슬롯 추출
    slots = extract_slots(question, candidates=candidates, llm_client=llm)

    # 화면에서 고른 값 반영. region/categories(필터)와 target_keywords(검색어 힌트)를 다르게 다룹니다:
    #   - 필터는 "덮어쓰기": 드롭다운에서 "금융"을 골랐는데 질문 문장에 "창업"이 있어서 둘 다 필터로 걸면
    #     terms 필터가 OR라 사용자가 고르지 않은 분야까지 섞여 나옵니다. 명시적으로 고른 값이 문장에서
    #     추측한 값보다 사용자의 의도에 가까우므로 그 값으로 바꿉니다.
    #   - 힌트는 "합치기": target_keywords는 결과를 걸러내지 않고 BM25 검색어에 보태기만 하므로(hybrid_search.py),
    #     둘 다 넣어도 결과가 사라질 위험이 없고 검색어가 풍부해질 뿐입니다.
    if region:
        slots["region"] = region
    if categories:
        slots["categories"] = list(categories)
    if target_keywords:
        # dict.fromkeys: 순서를 유지하면서 중복 제거 (set은 순서가 뒤섞임)
        slots["target_keywords"] = list(dict.fromkeys(slots["target_keywords"] + list(target_keywords)))

    # 5.2 하이브리드 검색
    chunks = hybrid_search(
        question,
        region=slots["region"],
        categories=slots["categories"],
        target_keywords=slots["target_keywords"],
        client=os_client,
    )
    trace["chunks"] = chunks

    # 5.6 [호출 전] 가드레일 - 관련 있는 근거가 없으면 LLM을 부르지 않음
    if not has_enough_evidence(chunks):
        return finish({"status": "no_evidence", "answer": NO_EVIDENCE_MESSAGE, "sources": [], "slots": slots})

    # 5.3 컨텍스트 조립
    context = build_context(chunks)

    if llm is None:
        # LLM이 없어도 검색은 됐으므로, 빈손으로 돌려보내지 않고 찾은 공고 목록은 보여줍니다.
        return finish({"status": "llm_unavailable", "answer": LLM_UNAVAILABLE_MESSAGE,
                       "sources": context["sources"], "slots": slots})

    # 5.4 + 5.5 프롬프트 구성 및 답변 생성
    try:
        raw_answer = generate_answer(question, context["context_text"], llm)
    except Exception as e:
        return finish({"status": "llm_error", "answer": LLM_UNAVAILABLE_MESSAGE,
                       "sources": context["sources"], "slots": slots, "error": str(e)})
    trace["raw_answer"] = raw_answer

    # 5.6 [호출 후] 가드레일 - 인용 검증
    checked = validate_answer(raw_answer, context["sources"])
    return finish({**checked, "slots": slots})


if __name__ == "__main__":
    client = get_client()
    candidates = fetch_candidate_values(client)
    for q in ["서울 소상공인 창업 자금 지원사업 알려줘", "오늘 날씨 어때"]:
        result = answer_question(q, os_client=client, candidates=candidates)
        print(f"\n[질문] {q}")
        print(f"  status : {result['status']}")
        print(f"  answer : {result['answer']}")
        print(f"  sources: {[s['program_name'][:30] for s in result['sources']]}")

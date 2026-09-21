# [WBS 5.6] 근거 없음 가드레일 로직 구현 (hallucination 방지)
#
# 프롬프트(prompt_template.py)로 "근거 밖의 말을 하지 마라"고 시키는 것만으로는 부족합니다 - 모델이
# 규칙을 어길 수 있기 때문입니다. 그래서 LLM 호출 "앞"과 "뒤"에 코드로 검사하는 관문을 하나씩 둡니다.
#
#   [호출 전] has_enough_evidence(): 검색 결과가 질문과 실제로 관련 있는지 점수로 확인.
#             관련 없으면 LLM을 아예 부르지 않고 고정 문구로 답함 (지어낼 기회 자체를 없앰 + 비용/시간 절약)
#   [호출 후] validate_answer(): 답변에 적힌 [번호] 인용이 실제 근거 문서 번호인지 확인.
#             인용이 하나도 없거나 없는 번호만 있으면 답변을 버리고 안전한 문구로 바꿈.

import re

from prompt_template import NO_EVIDENCE_MESSAGE

# ── [호출 전] 관련도 임계값 ───────────────────────────────────────────
# 왜 필요한가: kNN 검색은 "가장 가까운 k개"를 무조건 돌려줍니다. "오늘 날씨 어때" 같은 질문에도
# 지원사업 청크 5개가 나오고, 그걸 근거랍시고 LLM에 넘기면 억지 답변(hallucination)이 나옵니다.
#
# 왜 0.77인가 (2026-09-21 실측, 스크래치 스크립트로 hybrid_search()에 질의 10개를 넣어 확인):
#   관련 질의 5개("서울 소상공인 창업 자금", "여성기업 수출 지원" 등)의 최고 kNN 점수: 0.815 ~ 0.928
#   무관 질의 5개("오늘 날씨", "김치찌개 끓이는 법", "파이썬 리스트 정렬" 등)의 최고 kNN 점수: 0.656 ~ 0.722
#   -> 두 그룹 사이(0.722와 0.815)의 중간값 근처인 0.77로 설정. 이 표본에서는 10개 모두 올바르게 갈림.
#
# 왜 BM25 점수가 아니라 kNN 점수인가: 같은 실측에서 BM25는 무관 질의("손흥민 골 몇 개")도 4.12까지
#   나와서 관련 질의 최저값(5.92)과의 간격이 좁았습니다. kNN 쪽이 두 그룹을 더 확실하게 갈랐습니다.
#   또한 hybrid_search()의 score는 "후보군 안에서" min-max 정규화한 값이라 1등은 거의 항상 높게 나옵니다
#   - 절대적인 관련도를 볼 수 없으므로, 정규화 전 원본 kNN 점수(knn_score)를 씁니다.
#
# [한계] 질의 10개로 정한 잠정값입니다. WBS 7.1(테스트 질의셋 20~30개)이 만들어지면 그 데이터로
#   다시 조정해야 합니다. 너무 높이면 정상 질문도 "못 찾았다"고 답하고, 너무 낮추면 무관한 질문에 답합니다.
KNN_RELEVANCE_THRESHOLD = 0.77

# 인용이 하나도 없는 답변을 대체할 문구. NO_EVIDENCE_MESSAGE와 따로 두는 이유: "근거가 없어서 못 찾음"과
# "근거는 있었는데 모델이 인용 규칙을 어김"은 원인이 달라서, 로그/평가(WBS 7.3)에서 구분해야 합니다.
UNGROUNDED_MESSAGE = "답변의 근거를 확인할 수 없어 안내를 생략했습니다. 질문을 조금 더 구체적으로 바꿔서 다시 물어봐 주세요."

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")  # [1], [12] 같은 인용 표기에서 숫자만 뽑음


def has_enough_evidence(chunks: list[dict]) -> bool:
    """
    입력: hybrid_search()의 반환값
    출력: 검색 결과 중 원본 kNN 점수가 임계값 이상인 청크가 하나라도 있으면 True

    knn_score가 None인 청크(BM25에서만 잡힌 청크)는 판단에서 빠집니다. 전부 None이면 False가 되는데,
    이는 "의미상 가까운 문서가 후보 20개 안에도 없었다"는 뜻이라 근거 없음으로 보는 게 맞습니다.
    """
    knn_scores = [c["knn_score"] for c in chunks if c.get("knn_score") is not None]
    return bool(knn_scores) and max(knn_scores) >= KNN_RELEVANCE_THRESHOLD


def validate_answer(answer: str, sources: list[dict]) -> dict:
    """
    입력: answer - generator.generate_answer()의 원본 답변
          sources - context_builder.build_context()의 sources (citation_index 1..k)
    출력: {
        "status": "ok" | "no_evidence" | "ungrounded",
        "answer": 사용자에게 보여줄 최종 답변,
        "sources": 답변이 "실제로 인용한" 근거만 남긴 리스트,
        "invalid_citations": 근거 문서에 없는 번호 목록 (모델이 번호를 지어낸 흔적 - 로그/평가용),
    }
    """
    # 1) 모델이 스스로 "근거 없음" 문장을 낸 경우. 동일 비교(==)가 아니라 포함(in)으로 보는 이유:
    #    프롬프트 규칙 6("마지막 줄에 원문 확인 문구를 붙여라")과 겹쳐서 문장 뒤에 한 줄이 더 붙어
    #    나올 수 있기 때문입니다.
    if NO_EVIDENCE_MESSAGE in answer:
        return {"status": "no_evidence", "answer": NO_EVIDENCE_MESSAGE, "sources": [], "invalid_citations": []}

    cited = {int(n) for n in _CITATION_PATTERN.findall(answer)}
    valid_indexes = {s["citation_index"] for s in sources}
    valid_cited = cited & valid_indexes
    invalid_cited = sorted(cited - valid_indexes)

    # 2) 유효한 인용이 하나도 없으면, 답변 내용이 근거에서 왔다는 걸 확인할 방법이 없으므로 통째로 버림.
    #    (일부 문장이 맞을 수도 있지만, 어느 문장이 맞는지 가려낼 수 없어서 전부 버리는 쪽이 안전합니다 -
    #     Hallucination 비율 5% 이하 목표를 우선한 결정)
    if not valid_cited:
        return {"status": "ungrounded", "answer": UNGROUNDED_MESSAGE, "sources": [], "invalid_citations": invalid_cited}

    # 3) 유효한 인용이 있으면 답변은 살리되, 없는 번호([9] 등)는 지워서 사용자가 존재하지 않는 출처를
    #    클릭하는 일이 없게 합니다. 지운 번호는 invalid_citations로 남겨서 평가 때 확인할 수 있게 합니다.
    cleaned = answer
    for n in invalid_cited:
        cleaned = cleaned.replace(f"[{n}]", "")

    return {
        "status": "ok",
        "answer": cleaned,
        "sources": [s for s in sources if s["citation_index"] in valid_cited],
        "invalid_citations": invalid_cited,
    }

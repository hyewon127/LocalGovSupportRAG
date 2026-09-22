# [WBS 7.2~7.3] 평가 지표 계산 함수 모음 - 검색(Hit@k, MRR)과 답변(출처 인용, 근거 없는 수치) 지표
#
# 왜 계산 로직만 따로 뺐는가:
#   retrieval_eval.py / answer_eval.py는 OpenSearch·LLM을 불러야 해서 돌리는 데 시간이 걸리고 환경도 필요합니다.
#   반면 "순위 리스트와 정답이 주어지면 Hit@5가 얼마인가" 같은 계산은 순수 함수라서, 여기 모아두면
#   tests/test_evaluation_metrics.py에서 손으로 계산한 기대값과 바로 비교해 검증할 수 있습니다.
#   평가 스크립트에서 지표 계산이 틀리면 리포트의 모든 숫자가 틀려지므로, 계산부터 따로 검증하는 게 순서입니다.
#
# [순위(rank)의 기준 - 분석모델 정의서 2.5 "top-5 검색 결과에 정답 문서가 포함된 비율"]
#   hybrid_search()가 돌려주는 top-5는 "청크" 5개이고, 정답은 "사업(program_id)" 단위로 라벨링되어 있습니다.
#   그래서 순위는 "LLM에 넘어가는 청크 목록에서 해당 사업이 처음 등장한 위치"로 정의합니다.
#   예) 청크 순서가 [A, A, B, C, D]이고 정답이 B면 rank=3 (사업 기준으로 2등이지만 청크 기준으로 3번째).
#   같은 사업의 청크가 앞자리를 차지해 다른 사업을 밀어내는 것도 실제로 LLM이 보는 근거를 줄이는 것이므로,
#   그 효과까지 점수에 반영하려는 의도입니다 (hybrid_search.py MAX_CHUNKS_PER_PROGRAM 주석 참고).

import re


# ── 1. 검색 지표 (WBS 7.2) ───────────────────────────────────────────
def first_relevant_rank(retrieved_program_ids: list[str], relevant: set[str], k: int) -> int | None:
    """
    입력: retrieved_program_ids - 청크 순서대로 나열한 program_id (중복 가능, 위 [순위의 기준] 참고)
          relevant - 정답 program_id 집합
          k - 앞에서 몇 개까지 볼지
    출력: 정답이 처음 나온 순위(1부터 시작), 앞 k개 안에 없으면 None
    """
    for rank, program_id in enumerate(retrieved_program_ids[:k], start=1):
        if program_id in relevant:
            return rank
    return None


def hit_at_k(retrieved_program_ids: list[str], relevant: set[str], k: int) -> float:
    """앞 k개 안에 정답이 하나라도 있으면 1.0, 없으면 0.0 (질의 1개 기준 - 평균 내면 Hit@k 비율)"""
    return 1.0 if first_relevant_rank(retrieved_program_ids, relevant, k) is not None else 0.0


def reciprocal_rank(retrieved_program_ids: list[str], relevant: set[str], k: int) -> float:
    """
    1 / (첫 정답 순위). 1등이면 1.0, 2등이면 0.5, 3등이면 0.333... 앞 k개 안에 없으면 0.
    평균 내면 MRR@k. Hit@k가 "찾았냐"만 본다면, MRR은 "얼마나 위에서 찾았냐"까지 봅니다
    (분석모델 정의서 2.5: "정답 문서가 상위에 노출되는 정도", 목표 0.6 이상).
    """
    rank = first_relevant_rank(retrieved_program_ids, relevant, k)
    return 1.0 / rank if rank is not None else 0.0


def recall_at_k(retrieved_program_ids: list[str], relevant: set[str], k: int) -> float:
    """
    정답 사업 중 앞 k개 안에 들어온 비율. 정의서의 목표 지표는 아니지만 보조 지표로 봅니다.
    정답이 여러 개인 질의(예: 서울 자치구 융자 9건)에서 Hit@5는 1건만 찾아도 만점이라,
    "LLM이 선택지를 충분히 받았는가"는 Recall이 더 잘 보여줍니다.
    """
    if not relevant:
        return 0.0
    found = set(retrieved_program_ids[:k]) & relevant
    return len(found) / len(relevant)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ── 2. 답변 지표 (WBS 7.3) ───────────────────────────────────────────
_CITATION = re.compile(r"\[(\d+)\]")

# 답변 속 "사실로 검증 가능한 수치": 금액/비율/인원·기업 수/기간. 이 값들은 공고마다 다르고 틀리면 사용자에게
# 실제 피해(잘못된 지원금 기대, 마감 착오)를 주는 대표적인 hallucination 유형이라 따로 검사합니다.
# 쉼표/공백 차이("5,000만원" vs "5000만 원")는 비교 전에 없애서 형식 차이로 오탐하지 않게 합니다.
_FACT_NUMBER = re.compile(r"\d[\d,.]*\s*(?:억\s*원|만\s*원|천\s*원|억|만|원|%|개사|개월|명|년|일)")


def citation_stats(answer: str, num_sources: int) -> dict:
    """
    입력: answer - LLM 원본 답변(가드레일 적용 전), num_sources - 컨텍스트로 준 근거 개수(인용 번호 1..num_sources)
    출력: {"cited": 유효 인용 번호 목록, "invalid": 없는 번호 목록, "has_valid_citation": bool}
    출처 인용률(정의서 목표 95% 이상) = 유효 인용이 1개 이상 있는 답변의 비율로 계산합니다.
    """
    numbers = {int(n) for n in _CITATION.findall(answer)}
    valid = sorted(n for n in numbers if 1 <= n <= num_sources)
    invalid = sorted(n for n in numbers if not 1 <= n <= num_sources)
    return {"cited": valid, "invalid": invalid, "has_valid_citation": bool(valid)}


def _normalize(text: str) -> str:
    return re.sub(r"[\s,]", "", text)


def unsupported_numbers(answer: str, cited_contexts: list[str]) -> list[str]:
    """
    입력: answer - LLM 답변, cited_contexts - 답변이 인용한 근거들의 원문(LLM이 실제로 받은 텍스트)
    출력: 답변에 나왔지만 인용한 근거 어디에도 없는 수치 목록 (= 근거 없이 지어냈을 가능성이 있는 수치)

    [hallucination 대리 지표(proxy)인 이유와 한계]
      정확한 hallucination 판정은 사람이 문장마다 근거와 대조하거나 LLM 심판(LLM-as-judge)을 써야 하는데,
      둘 다 비용이 들거나(무과금 방침) 재현성이 떨어집니다. 대신 "수치"만큼은 문자열 비교로 기계적으로
      검사할 수 있어서 이걸 대리 지표로 씁니다.
      - 놓치는 것: 수치가 없는 지어낸 문장(예: 존재하지 않는 자격 요건) -> 실제 비율보다 낮게 나올 수 있음
      - 잘못 잡는 것: 근거의 표기와 다른 형식(예: "3천만원" vs "30,000,000원") -> 실제보다 높게 나올 수 있음
      리포트에는 이 한계를 같이 적어야 합니다.
    """
    context = _normalize(" ".join(cited_contexts))
    found = []
    for match in _FACT_NUMBER.finditer(answer):
        token = _normalize(match.group())
        if token not in context:
            found.append(match.group().strip())
    return found

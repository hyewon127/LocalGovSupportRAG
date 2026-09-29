# [WBS 5.5] LLM 호출 및 답변 생성 함수 구현
#
# 역할을 "LLM 한 번 호출해서 답변 문자열을 받아오는 것"으로만 좁혔습니다.
#   - 어떤 규칙으로 답하게 할지      -> prompt_template.py (5.4)
#   - 답변이 규칙을 지켰는지 검사     -> guardrail.py (5.6)
#   - 검색부터 응답까지 순서대로 묶기 -> pipeline.py
# 이렇게 나눠두면 나중에 LLM 업체를 바꾸거나(llm_client.py) 프롬프트만 고쳐서 평가(WBS 7)를 다시 돌릴 때
# 이 파일은 손대지 않아도 됩니다.

from openai import OpenAI

from llm_client import LLM_MODEL, LLM_TEMPERATURE
from prompt_template import build_messages

# 답변 길이 상한. top-5 사업을 사업별로 몇 줄씩 요약하면 보통 수백 토큰이면 충분하고,
# 상한이 없으면 모델이 근거 문서 본문을 통째로 되풀이하는 경우가 있어 응답 시간 목표
# (분석모델 정의서 2.5: 평균 응답시간 5초 이내)를 넘기기 쉽습니다.
MAX_ANSWER_TOKENS = 800


def generate_answer(question: str, context_text: str, client: OpenAI) -> str:
    """
    입력: question - 사용자 원문 질의
          context_text - context_builder.build_context()의 context_text
          client - llm_client.get_llm_client()가 만든 클라이언트 (None이 아닌 것이 보장된 상태로 호출)
    출력: 모델이 생성한 답변 문자열 (앞뒤 공백 제거)

    예외를 여기서 잡지 않는 이유: 네트워크 오류/키 오류가 났을 때 "빈 답변"을 돌려주면, 호출하는 쪽에서
    "모델이 빈 답을 냈다"와 "호출 자체가 실패했다"를 구분할 수 없습니다. pipeline.py가 예외를 받아서
    상태값(status)으로 구분해 응답합니다.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=build_messages(question, context_text),
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    # content가 None으로 오는 경우(안전 필터 등으로 응답이 막힌 경우)가 있어 빈 문자열로 바꿔둡니다.
    # 빈 답변은 guardrail.py에서 "인용 없음"으로 걸러집니다.
    return (response.choices[0].message.content or "").strip()

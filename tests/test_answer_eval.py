# [WBS 7.3] 답변 평가(answer_eval.evaluate_answers) 코드 경로 검증 - 가짜 LLM으로 실제 파이프라인을 끝까지 통과시킨다
#
# 실행 (OpenSearch 실행 중, 프로젝트 루트에서):
#   .venv\Scripts\python.exe tests\test_answer_eval.py
#
# [왜 필요한가] 출처 인용률/hallucination 측정은 LLM 키가 있어야 실제로 돌아가는데, 지금은 키가 없습니다.
#   키를 넣은 날 처음 실행했는데 코드가 터지면 곤란하므로, "LLM 응답만 가짜이고 나머지(슬롯 추출 -> 검색 ->
#   가드레일 -> 프롬프트 -> 인용/수치 검사)는 전부 진짜"인 상태로 미리 돌려봅니다.
#   가짜 LLM은 실제로 받은 프롬프트에서 1번 근거의 지원금액을 읽어 인용하고, 근거에 없는 금액을 일부러 하나 섞습니다.
#   -> 평가 코드가 "인용 있음"과 "지어낸 수치 1개"를 정확히 잡아내는지 확인할 수 있습니다.

import re
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

from answer_eval import evaluate_answers  # noqa: E402  (answer_eval이 src/rag, src/indexing 경로를 잡아줌)
from config import get_client  # noqa: E402
from query_slots import fetch_candidate_values  # noqa: E402

FAKE_AMOUNT = "9,876억원"  # 어떤 공고에도 없을 금액 (지어낸 수치 역할)
results: list[tuple[str, bool]] = []


def check(name: str, condition, detail="") -> None:
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""))


class PromptReadingFakeLLM:
    """
    받은 프롬프트의 [1]번 근거에서 지원금액을 읽어 답하는 가짜 LLM (chat.completions.create만 흉내).

    호출이 두 종류라는 점에 주의: 정규식으로 슬롯을 못 찾은 질의는 query_slots.py가 슬롯 추출용으로도 LLM을
    부릅니다(tools 인자가 붙은 호출). 처음엔 "답변 생성 2번"만 예상했다가 호출이 4번 나와서 알게 된 실제 동작이라,
    두 종류를 따로 셉니다. 슬롯 추출 호출에는 "추출 결과 없음"(tool_calls 없음)으로 답해 정규식 결과로 폴백시킵니다.
    """

    def __init__(self):
        self.chat = self
        self.completions = self
        self.answer_calls = 0
        self.slot_calls = 0

    def create(self, messages, **kwargs):
        if "tools" in kwargs:
            self.slot_calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None))])
        self.answer_calls += 1
        user = messages[-1]["content"]
        amount = re.search(r"- 지원금액: (.+)", user).group(1).strip()
        content = f"첫 번째 사업은 지원금액이 {amount}입니다 [1]. 별도로 최대 {FAKE_AMOUNT}까지 지원됩니다 [1]."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))])


def main() -> int:
    client = get_client()
    candidates = fetch_candidate_values(client)
    queries = [
        {"id": "T1", "type": "on_topic", "question": "소상공인 라이브커머스 제작 지원사업 알려줘"},
        {"id": "T2", "type": "on_topic", "question": "해외규격인증 획득 비용 지원받을 수 있나요"},
        {"id": "T3", "type": "off_topic", "question": "감기 빨리 낫는 방법 알려줘"},
    ]
    llm = PromptReadingFakeLLM()
    out = evaluate_answers(queries, client, candidates, llm)
    s = out["summary"]
    by_id = {r["id"]: r for r in out["per_query"]}

    check("관련 질의 2개는 LLM까지 도달", by_id["T1"]["generated"] and by_id["T2"]["generated"])
    check("무관 질의는 가드레일에서 차단(LLM 미호출)", not by_id["T3"]["generated"] and by_id["T3"]["final_status"] == "no_evidence")
    check("답변 생성 LLM 호출 = 2 (관련 질의 수)", llm.answer_calls == 2, llm.answer_calls)
    # 슬롯 추출 폴백은 정규식이 아무것도 못 찾은 질의에서만 호출 (몇 번인지는 질의 문장에 따라 달라서 기록만 함)
    print(f"   참고: 슬롯 추출 LLM 폴백 호출 {llm.slot_calls}회")
    check("출처 인용률 100% ([1] 인용)", s["citation_rate"] == 1.0, s["citation_rate"])
    check("지어낸 금액을 근거 없는 수치로 검출", all(FAKE_AMOUNT in r["unsupported_numbers"]
                                        for r in out["per_query"] if r["generated"]),
          [r.get("unsupported_numbers") for r in out["per_query"]])
    check("근거에 있는 금액은 검출하지 않음", all(len(r["unsupported_numbers"]) == 1
                                      for r in out["per_query"] if r["generated"]))
    check("hallucination 대리 지표 100% (모든 답변에 지어낸 수치)", s["hallucination_proxy_rate"] == 1.0)
    check("무관 질의 올바른 거절 100%", s["off_topic_correct_refusal"] == 1.0)
    check("최종 상태: 인용이 있으니 가드레일 통과(ok)", by_id["T1"]["final_status"] == "ok", by_id["T1"]["final_status"])
    print(f"   예시 답변: {by_id['T1']['raw_answer']}")

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

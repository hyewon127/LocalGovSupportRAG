# [2026-09-29 로컬 LLM(Ollama) 연결 중 발견한 문제들의 재현 테스트]
#
# 실행 (OpenSearch·Ollama 없이 돌아감, 프로젝트 루트에서, 몇 초):
#   .venv\Scripts\python.exe tests\test_llm_provider.py
#
# 확인하는 것 (모두 실제로 겪은 문제):
#   1. LLM 자동 선택 - .env에 OPENAI_API_KEY만 있는데 기본값 upstage의 키 칸만 봐서 "키 없음"이 됐던 문제.
#      지금 순서: 직접 지정 > upstage 키 > ollama(모델 있음) > openai 키 > upstage(비활성)
#   2. 바꿔 쓴 거절 문장 - 7B 모델이 고정 문장 대신 "OO을 찾지 못했습니다"/"찾을 수 없습니다"로 거절하면
#      "인용 규칙 위반(ungrounded)"으로 잘못 분류되던 문제. 단, 인용이 있으면 일부라도 안내한 답변이므로 거절로 보면 안 됨.
#   3. Ollama 슬롯 추출 차단 - 처음에는 "인자 생략"만 막아서, 실제 클라이언트를 직접 넘기는 pipeline.py에서는
#      한 번도 막히지 않았던 문제(평가 수치가 전혀 안 바뀌어서 발견).

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "indexing"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))

from openai import OpenAI  # noqa: E402

import llm_client  # noqa: E402
import query_slots  # noqa: E402
from guardrail import UNGROUNDED_MESSAGE, validate_answer  # noqa: E402
from prompt_template import NO_EVIDENCE_MESSAGE  # noqa: E402

results: list[bool] = []


def check(name: str, ok: bool, detail=None) -> None:
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok or detail is None else f"  ({detail})"))


# ── 1. LLM 자동 선택 ─────────────────────────────────────────────
def pick(env: dict, ollama_models: list[str]) -> str:
    """env의 키들만 설정된 상태에서 _pick_provider()를 부름. Ollama 서버 응답은 가짜 목록으로 바꿔 끼움."""
    names = ["LLM_PROVIDER", "UPSTAGE_API_KEY", "OPENAI_API_KEY"]
    saved = {n: os.environ.get(n) for n in names}
    original = llm_client._ollama_models
    try:
        for n in names:
            os.environ.pop(n, None)
        os.environ.update(env)
        llm_client._ollama_models = lambda: ollama_models
        return llm_client._pick_provider()
    finally:
        llm_client._ollama_models = original
        for n, v in saved.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


def test_provider_selection() -> None:
    print("\n── 1. LLM 자동 선택 ──")
    rag = llm_client.OLLAMA_RAG_MODEL
    check("OpenAI 키만 있으면 openai (예전에는 upstage 칸만 보고 '키 없음')",
          pick({"OPENAI_API_KEY": "sk-real"}, []) == "openai")
    check("Ollama 모델이 있으면 유료 openai보다 ollama 우선 (무과금 방침)",
          pick({"OPENAI_API_KEY": "sk-real"}, [rag]) == "ollama")
    check("upstage 키가 있으면 upstage 최우선 (분석모델 정의서의 선택)",
          pick({"UPSTAGE_API_KEY": "up-real", "OPENAI_API_KEY": "sk-real"}, [rag]) == "upstage")
    check("예시 값('your...') 키는 없는 것으로 봄", pick({"OPENAI_API_KEY": "your_key_here"}, []) == "upstage")
    check("Ollama가 켜져 있어도 쓸 모델이 없으면 선택 안 함", pick({}, ["llama3:8b"]) == "upstage")
    check("LLM_PROVIDER를 직접 적으면 그대로 따름", pick({"LLM_PROVIDER": "none", "OPENAI_API_KEY": "sk-real"}, [rag]) == "none")
    check("Ollama 모델은 한도를 늘린 -rag 버전 우선",
          llm_client._pick_ollama_model(["qwen2.5:7b", rag]) == rag)


# ── 2. 바꿔 쓴 거절 문장 ──────────────────────────────────────────
def test_refusal_variants() -> None:
    print("\n── 2. 거절 문장 인식 ──")
    sources = [{"citation_index": 1, "program_id": "P1"}, {"citation_index": 2, "program_id": "P2"}]
    cases = [
        (NO_EVIDENCE_MESSAGE, "no_evidence", "고정 문장 그대로"),
        ("제공된 공고 자료에서 서울 소상공인 창업 자금 지원사업을 찾지 못했습니다.", "no_evidence", "질문을 끼워 넣은 변형(실제 사례)"),
        ("제공된 공고 자료에서 개인 주택담보대출 금리에 대한 정보를 찾을 수 없습니다.", "no_evidence", "'찾을 수 없습니다' 변형(실제 사례)"),
        ("A 사업은 소상공인 대상입니다 [1]. B 관련 사업은 찾지 못했습니다.", "ok", "인용이 있으면 일부 안내한 답변 -> 거절 아님"),
        ("A 사업은 소상공인 대상입니다.", "ungrounded", "거절도 인용도 없으면 여전히 규칙 위반"),
    ]
    for answer, expected, label in cases:
        result = validate_answer(answer, sources)
        check(f"{label} -> {expected}", result["status"] == expected, result["status"])
    result = validate_answer("A 사업은 소상공인 대상입니다.", sources)
    check("규칙 위반 시 안내 문구는 UNGROUNDED_MESSAGE", result["answer"] == UNGROUNDED_MESSAGE)


# ── 3. 슬롯 추출 LLM 차단 ─────────────────────────────────────────
def test_slot_llm_block() -> None:
    print("\n── 3. 슬롯 추출 LLM 차단 (slot_llm=False 공급자) ──")
    query = "요식업 하는데 받을 수 있는 지원금 있나요"  # 정규식이 아무것도 못 잡는 문장 = LLM 폴백 경로
    original = query_slots.SLOT_LLM_ENABLED
    query_slots.SLOT_LLM_ENABLED = False
    # 실제로 요청이 나가면 연결 실패 예외 -> extract_slots가 흡수하고 경고를 찍음. 호출 여부는 create를 바꿔 끼워 기록.
    real_client = OpenAI(api_key="test", base_url="http://127.0.0.1:9/v1")
    calls = []
    real_client.chat.completions.create = lambda **kw: calls.append(kw) or (_ for _ in ()).throw(RuntimeError("called"))

    class FakeLLM:  # 테스트용 가짜 LLM (OpenAI 객체가 아님) - 슬롯 추출 코드 경로를 검사하려는 의도이므로 허용돼야 함
        def __init__(self):
            self.chat = self
            self.completions = self
            self.used = False

        def create(self, **kwargs):
            self.used = True
            raise RuntimeError("fake")

    try:
        slots = query_slots.extract_slots(query, llm_client=real_client)
        check("실제 클라이언트를 직접 넘겨도 LLM 호출 안 함 (pipeline.py 경로)", not calls and slots["categories"] == [], calls)
        fake = FakeLLM()
        query_slots.extract_slots(query, llm_client=fake)
        check("가짜 LLM은 그대로 호출됨 (테스트 의도 존중)", fake.used)
    finally:
        query_slots.SLOT_LLM_ENABLED = original
    check("Ollama 설정은 slot_llm=False", llm_client._LLM_SPECS["ollama"]["slot_llm"] is False)


def main() -> int:
    test_provider_selection()
    test_refusal_variants()
    test_slot_llm_block()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

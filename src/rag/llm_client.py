# [WBS 5 공통] LLM(답변 생성/슬롯 추출) 호출 설정 모음
#
# 왜 별도 파일로 뺐는가: src/indexing/config.py가 EMBEDDING_PROVIDER 하나로 임베딩 모델을 갈아끼우게
# 만든 것과 같은 이유입니다. LLM을 쓰는 곳이 두 군데(query_slots.py의 슬롯 추출 폴백, generator.py의
# 답변 생성)라서, 모델명/접속 주소를 각 파일에 따로 적어두면 모델을 바꿀 때 한 군데를 빠뜨리기 쉽습니다.
#
# [결정 - 분석모델 정의서.pptx 2.4 "LLM 모델 후보 비교"를 그대로 따름]
#   채택(✔): Upstage Solar Pro - 한국어 특화, 국내 서비스 친화적
#   비교군 : OpenAI GPT 계열  - 안정적 성능, API 호출 비용 발생
#   선정 기준에 "무과금 운영 (로컬 임베딩 + 무료 티어 LLM만 사용, 유료 API 호출 없음)"이 명시돼 있고,
#   2026-08-21에 "무과금 운영 방침에 따라 임베딩/LLM 모두 무료 대안으로 조기 확정"했다고 적혀 있습니다.
#   -> 기본값은 upstage. (처음 만든 query_slots.py가 OpenAI를 쓰고 있었는데, 이 문서 결정과 어긋나서
#      이 파일로 통일하면서 바로잡았습니다.)
#
# Upstage API는 OpenAI API와 요청/응답 형식이 같은 "OpenAI 호환" API라서, openai 패키지를 그대로 쓰고
# base_url만 Upstage 주소로 바꾸면 됩니다 -> 라이브러리를 새로 설치할 필요가 없고, 나중에 OpenAI로
# 바꾸고 싶으면 .env의 LLM_PROVIDER만 openai로 바꾸면 됩니다 (나머지 코드는 그대로).
# (2026-09-21 확인: https://api.upstage.ai/v1/chat/completions 에 가짜 키로 요청 -> 404가 아니라 401
#  "Your API key is invalid"가 돌아옴 = 주소는 맞고 키만 없는 상태)
#
# [결정 변경 2026-09-29 - 로컬 LLM(Ollama) 추가]
#   Upstage 키는 끝내 발급하지 않았고, .env의 OpenAI 키는 인증은 되지만 잔액이 없어 429(insufficient_quota)로
#   거절됐습니다. 정의서의 "무과금 운영" 방침을 지키면서 실제 답변을 만들 수 있는 방법으로 Ollama를 추가했습니다.
#   Ollama도 OpenAI 호환 API(/v1)를 제공하므로 이 파일에 설정 한 덩어리만 추가하면 되고, generator.py·query_slots.py는
#   그대로입니다. 덤으로 인터넷 없이도 시연할 수 있습니다.

import json
import os
import urllib.request

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_LLM_SPECS = {
    "upstage": {
        "base_url": "https://api.upstage.ai/v1",
        "api_key_env": "UPSTAGE_API_KEY",   # https://console.upstage.ai/api-keys 에서 발급
        "default_model": "solar-pro2",
        "timeout_sec": 20,
        "max_retries": 1,
        "slot_llm": True,
    },
    "openai": {
        "base_url": None,                   # None이면 openai 패키지 기본 주소(api.openai.com) 사용
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "timeout_sec": 20,
        "max_retries": 1,
        "slot_llm": True,
    },
    "ollama": {
        # localhost가 아니라 127.0.0.1 - Windows에서 localhost는 IPv6(::1)를 먼저 시도해 요청마다 약 2초 늦어짐 (src/api/main.py 주석)
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") + "/v1",
        "api_key_env": None,                # 로컬 서버라 키가 없음. openai 패키지가 빈 키를 거부해서 아래에서 아무 문자열이나 넣음
        "default_model": None,              # 받아둔 모델 중에서 고름 (_pick_ollama_model)
        # 클라우드 API와 달리 내 PC GPU로 생성하므로 느림. 첫 호출은 모델을 GPU에 올리는 시간까지 더해짐.
        # 재시도 0번: 로컬 서버가 시간 초과면 다시 보내도 똑같이 느릴 뿐이라 기다리는 시간만 2배가 됨.
        "timeout_sec": 60,
        "max_retries": 0,
        # 슬롯 추출(query_slots.py) LLM 폴백을 쓰지 않음. 2026-09-29 평가셋 실측:
        #   qwen2.5:7b는 tool_choice="required"를 무시하고 대부분 일반 문장으로 답했고(도구 호출 실패), 가끔 호출하면
        #   분야를 틀리게 뽑았음(라이브커머스 질문 -> "창업, 금융"). 틀린 필터가 정답 공고를 걸러내서, 표현을 바꾼 정상 질문의
        #   가드레일 통과율이 81.5%(LLM 없음) -> 55.6%로 오히려 떨어짐. 폴백은 정규식이 아무것도 못 찾은 질문(= 표현을 바꾼
        #   질문)에서만 불리므로 피해가 정확히 실제 사용자 표현에 집중됨. 필터 없이 검색하는 쪽이 낫다고 판단.
        "slot_llm": False,
    },
    # LLM을 명시적으로 끔 (LLM_PROVIDER=none). tests/run_all.py가 띄우는 백엔드용 - 실제 LLM은 같은 질문에도 답/거절이
    # 바뀔 수 있어서 화면 테스트가 통과/실패를 오락가락하게 됨. 코드 회귀 테스트는 LLM 없이, 답변 품질은 answer_eval.py로 분리.
    "none": {
        "base_url": None,
        "api_key_env": None,
        "default_model": None,
        "timeout_sec": 20,
        "max_retries": 0,
        "slot_llm": False,
    },
}

# Ollama에서 쓸 모델 우선순위. qwen2.5를 고른 이유: 한국어 답변 가능 + query_slots.py가 쓰는 도구 호출(tool calling)을
# Ollama에서 지원 + 7b는 이 PC GPU(16GB)에서 빠름.
# "-rag"가 붙은 모델이 먼저인 이유: 원본(qwen2.5:7b)은 한 번에 읽는 양이 4096토큰이라 근거 문서가 잘려 나갑니다
# (ollama/Modelfile 주석의 실측: 5,800토큰 중 2,050토큰만 들어감). -rag는 그 한도를 16384로 늘린 버전입니다.
# 원본은 -rag가 없을 때의 대비용으로만 남겨두고, 쓰게 되면 get_llm_client가 경고를 띄웁니다.
# 다른 모델을 쓰고 싶으면 .env에 LLM_MODEL=모델명 (LLM_MODEL이 항상 우선).
OLLAMA_RAG_MODEL = "qwen2.5-rag:7b"
OLLAMA_PREFERRED_MODELS = [OLLAMA_RAG_MODEL, "qwen2.5:7b"]


def _is_real_key(value: str | None) -> bool:
    """비어 있거나 "your_key_here" 같은 예시 값이면 False (아래 get_llm_client 주석의 2026-09-21 사례)."""
    value = (value or "").strip()
    return bool(value) and not value.lower().startswith("your")


def _ollama_models() -> list[str]:
    """
    출력: Ollama에 받아둔 모델 이름 목록. Ollama가 꺼져 있으면 빈 리스트.

    import 시점에 한 번 불리므로 timeout을 1초로 짧게 둡니다 - Ollama가 없는 PC에서 서버 시작이 늦어지면 안 됨.
    requests 대신 표준 라이브러리 urllib를 쓰는 이유: 이 파일이 쓰는 외부 패키지를 늘리지 않으려고.
    """
    url = _LLM_SPECS["ollama"]["base_url"].removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return [m["name"] for m in json.load(response).get("models", [])]
    except (OSError, ValueError):  # 연결 거부/시간 초과(URLError는 OSError의 하위) 또는 JSON 이상
        return []


def _pick_ollama_model(available: list[str]) -> str | None:
    """받아둔 모델 중 OLLAMA_PREFERRED_MODELS 순서로 첫 번째. 하나도 없으면 None (Ollama를 쓰지 않음)."""
    return next((m for m in OLLAMA_PREFERRED_MODELS if m in available), None)


def _pick_provider() -> str:
    """
    출력: 사용할 LLM 제공사 이름 ("upstage" / "ollama" / "openai" / "none")

    [2026-09-29] 예전에는 LLM_PROVIDER가 없으면 무조건 "upstage"였습니다. 그래서 .env에 OPENAI_API_KEY만
    넣은 경우 키가 멀쩡히 있는데도 UPSTAGE_API_KEY 칸만 보고 "키 없음"으로 판단했습니다 (사용자는 키를 넣었는데
    챗봇은 계속 "LLM 비활성" - 원인이 로그 한 줄에만 있어서 알아채기 어려웠음).
    그래서 순서를 이렇게 정합니다:
      1) LLM_PROVIDER를 직접 적었으면 그대로 따름 (명시한 설정이 항상 우선)
      2) 안 적었으면 "쓸 수 있는 것" 중에서 무과금 방침 순서로 고름:
         upstage(키 있음, 정의서의 선택) -> ollama(켜져 있고 모델이 있음, 무료) -> openai(키 있음, 유료라 마지막)
      3) 아무것도 없으면 upstage (get_llm_client가 "키 없음"을 안내하고 LLM 없이 동작)
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if _is_real_key(os.getenv("UPSTAGE_API_KEY")):
        return "upstage"
    if _pick_ollama_model(_ollama_models()):
        return "ollama"
    if _is_real_key(os.getenv("OPENAI_API_KEY")):
        return "openai"
    return "upstage"


LLM_PROVIDER = _pick_provider()
_SPEC = _LLM_SPECS[LLM_PROVIDER]

# 모델명은 서비스 쪽에서 버전이 자주 바뀌므로(예: solar-pro -> solar-pro2) 코드 수정 없이
# .env의 LLM_MODEL로 덮어쓸 수 있게 했습니다. 에러가 나면 콘솔에서 현재 모델명을 확인해서 넣으세요.
LLM_MODEL = (os.getenv("LLM_MODEL")
             or (_pick_ollama_model(_ollama_models()) if LLM_PROVIDER == "ollama" else _SPEC["default_model"]))

# 분석모델 정의서 1.3 "LLM temperature" 권장값 0.2 (탐색 범위 0~0.5).
# temperature가 낮을수록 같은 질문에 같은 답을 내고 "지어내기"가 줄어듭니다 - 공고문 내용을
# 정확히 옮겨야 하는 이 서비스에서는 창의성보다 사실성이 중요하므로 낮게 둡니다.
LLM_TEMPERATURE = 0.2

# [WBS 8.1] 호출 1번의 최대 대기 시간과 실패 시 재시도 횟수 (값은 위 _LLM_SPECS에 공급자별로 있음).
#   openai 패키지 기본값은 timeout 600초 + 재시도 2번이라, LLM 서버가 응답 없이 멈추면 요청 하나가 최대
#   30분 동안 서버 스레드를 붙잡습니다. 그동안 화면(Streamlit)은 이미 타임아웃으로 포기했는데 서버만 계속 기다리는 셈입니다.
#   클라우드 API(upstage/openai)는 답변 800토큰(generator.py MAX_ANSWER_TOKENS)을 보통 수 초에 끝내므로 20초 + 재시도 1번,
#   로컬 Ollama는 60초 + 재시도 0번입니다.
#   -> 최악의 경우 슬롯 추출 폴백 + 답변 생성 = 클라우드 20x2 + 20x2 = 80초, Ollama 60 + 60 = 120초.
#      ui/api_client.py의 CHAT_TIMEOUT_SEC(150초)가 이보다 길어야 화면이 먼저 끊지 않습니다 (둘을 바꿀 때는 같이 봐야 함).
LLM_TIMEOUT_SEC = _SPEC["timeout_sec"]
SLOT_LLM_ENABLED = _SPEC["slot_llm"]  # 위 ollama 설정의 slot_llm 주석 참고
LLM_MAX_RETRIES = _SPEC["max_retries"]

# [WBS 8.4] "LLM 클라이언트를 안 넘겼음"을 나타내는 표시값(sentinel).
#   예전에는 pipeline.answer_question()/query_slots.extract_slots()가 llm_client=None을 "안 넘겼으니 알아서 만들어라"로
#   해석해서, 호출하는 쪽이 "LLM 없이 해라"라는 뜻으로 None을 넘겨도 .env에 키가 있으면 get_llm_client()로 다시 만들어
#   LLM을 불렀습니다. 지금은 키가 없어서 드러나지 않았지만, 키를 넣는 순간 테스트(가짜 LLM 대신 None으로 "LLM 없음"
#   경로를 검사하는 곳)가 실제 유료 API를 호출하게 되는 잠재 버그였습니다. 그래서 두 뜻을 분리합니다:
#     인자를 생략(= USE_DEFAULT_LLM) -> .env 설정대로 클라이언트를 만듦
#     None을 명시                    -> LLM을 쓰지 않음
USE_DEFAULT_LLM = object()


def get_llm_client() -> OpenAI | None:
    """
    출력: LLM 클라이언트. Ollama는 받아둔 모델이 없으면, 나머지는 키가 없거나 "your_key_here" 같은 자리표시자면 None.

    None을 돌려주는 이유: 키가 없을 때 여기서 예외를 던지면, 이 모듈을 import만 해도 전체
    파이프라인이 죽습니다. 검색(BM25/kNN)은 LLM 없이도 동작하므로, 호출하는 쪽에서 None을 보고
    "LLM 없이 할 수 있는 만큼만" 하도록 맡깁니다.

    자리표시자 검사를 하는 이유: 2026-09-21에 .env의 OPENAI_API_KEY가 실제 키가 아니라 "your_..."로
    시작하는 예시 값이었는데, 값이 비어 있지 않아서 "설정됨"으로 보였고 실제 요청을 보낸 뒤에야
    401로 드러났습니다. 요청을 보내기 전에 걸러내면 원인이 더 빨리 보입니다.
    """
    if LLM_PROVIDER == "none":
        print("[LLM 비활성] LLM_PROVIDER=none - LLM 없이 공고 목록만 안내합니다.")
        return None
    if LLM_PROVIDER == "ollama":
        if not LLM_MODEL:
            print(f"[LLM 비활성] Ollama에 쓸 모델이 없습니다. `ollama pull {OLLAMA_PREFERRED_MODELS[0]}`로 받으세요.")
            return None
        if LLM_MODEL != OLLAMA_RAG_MODEL and not os.getenv("LLM_MODEL"):
            print(f"[LLM 경고] {LLM_MODEL}은 한 번에 4096토큰만 읽어서 근거 문서가 잘립니다. "
                  f"`ollama create {OLLAMA_RAG_MODEL} -f ollama/Modelfile`로 만들어 쓰세요.")
        # 키는 검사하지 않지만 openai 패키지가 빈 문자열을 거부하므로 아무 값("ollama")이나 넣음
        return OpenAI(api_key="ollama", base_url=_SPEC["base_url"], timeout=LLM_TIMEOUT_SEC, max_retries=LLM_MAX_RETRIES)

    api_key = os.getenv(_SPEC["api_key_env"], "").strip()
    if not _is_real_key(api_key):
        print(f"[LLM 비활성] .env에 {_SPEC['api_key_env']}가 없거나 예시 값입니다 (LLM_PROVIDER={LLM_PROVIDER}).")
        return None
    return OpenAI(api_key=api_key, base_url=_SPEC["base_url"], timeout=LLM_TIMEOUT_SEC, max_retries=LLM_MAX_RETRIES)


def resolve_llm_client(llm_client):
    """
    입력: 호출하는 쪽이 넘긴 llm_client 인자 (USE_DEFAULT_LLM / None / 클라이언트 객체)
    출력: 실제로 쓸 클라이언트 또는 None. 위 USE_DEFAULT_LLM 주석의 규칙을 한 곳에서 적용합니다.
    """
    return get_llm_client() if llm_client is USE_DEFAULT_LLM else llm_client

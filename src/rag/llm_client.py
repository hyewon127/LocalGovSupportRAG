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

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_LLM_SPECS = {
    "upstage": {
        "base_url": "https://api.upstage.ai/v1",
        "api_key_env": "UPSTAGE_API_KEY",   # https://console.upstage.ai/api-keys 에서 발급
        "default_model": "solar-pro2",
    },
    "openai": {
        "base_url": None,                   # None이면 openai 패키지 기본 주소(api.openai.com) 사용
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "upstage")
_SPEC = _LLM_SPECS[LLM_PROVIDER]

# 모델명은 서비스 쪽에서 버전이 자주 바뀌므로(예: solar-pro -> solar-pro2) 코드 수정 없이
# .env의 LLM_MODEL로 덮어쓸 수 있게 했습니다. 에러가 나면 콘솔에서 현재 모델명을 확인해서 넣으세요.
LLM_MODEL = os.getenv("LLM_MODEL") or _SPEC["default_model"]

# 분석모델 정의서 1.3 "LLM temperature" 권장값 0.2 (탐색 범위 0~0.5).
# temperature가 낮을수록 같은 질문에 같은 답을 내고 "지어내기"가 줄어듭니다 - 공고문 내용을
# 정확히 옮겨야 하는 이 서비스에서는 창의성보다 사실성이 중요하므로 낮게 둡니다.
LLM_TEMPERATURE = 0.2

# [WBS 8.1] 호출 1번의 최대 대기 시간과 실패 시 재시도 횟수.
#   openai 패키지 기본값은 timeout 600초 + 재시도 2번이라, LLM 서버가 응답 없이 멈추면 요청 하나가 최대
#   30분 동안 서버 스레드를 붙잡습니다. 그동안 화면(Streamlit)은 이미 타임아웃으로 포기했는데 서버만 계속 기다리는 셈입니다.
#   답변 800토큰(generator.py MAX_ANSWER_TOKENS) 생성은 보통 수 초면 끝나므로 20초면 넉넉하고,
#   재시도 1번은 일시적인 네트워크 끊김 정도만 구제합니다.
#   -> 최악의 경우 슬롯 추출 폴백(20초x2) + 답변 생성(20초x2) = 80초. ui/api_client.py의 CHAT_TIMEOUT_SEC(90초)가
#      이보다 길어야 화면이 먼저 끊지 않습니다 (둘을 바꿀 때는 같이 봐야 함).
LLM_TIMEOUT_SEC = 20
LLM_MAX_RETRIES = 1

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
    출력: LLM 클라이언트. 키가 없거나 "your_key_here" 같은 자리표시자면 None.

    None을 돌려주는 이유: 키가 없을 때 여기서 예외를 던지면, 이 모듈을 import만 해도 전체
    파이프라인이 죽습니다. 검색(BM25/kNN)은 LLM 없이도 동작하므로, 호출하는 쪽에서 None을 보고
    "LLM 없이 할 수 있는 만큼만" 하도록 맡깁니다.

    자리표시자 검사를 하는 이유: 2026-09-21에 .env의 OPENAI_API_KEY가 실제 키가 아니라 "your_..."로
    시작하는 예시 값이었는데, 값이 비어 있지 않아서 "설정됨"으로 보였고 실제 요청을 보낸 뒤에야
    401로 드러났습니다. 요청을 보내기 전에 걸러내면 원인이 더 빨리 보입니다.
    """
    api_key = os.getenv(_SPEC["api_key_env"], "").strip()
    if not api_key or api_key.lower().startswith("your"):
        print(f"[LLM 비활성] .env에 {_SPEC['api_key_env']}가 없거나 예시 값입니다 (LLM_PROVIDER={LLM_PROVIDER}).")
        return None
    return OpenAI(api_key=api_key, base_url=_SPEC["base_url"], timeout=LLM_TIMEOUT_SEC, max_retries=LLM_MAX_RETRIES)


def resolve_llm_client(llm_client):
    """
    입력: 호출하는 쪽이 넘긴 llm_client 인자 (USE_DEFAULT_LLM / None / 클라이언트 객체)
    출력: 실제로 쓸 클라이언트 또는 None. 위 USE_DEFAULT_LLM 주석의 규칙을 한 곳에서 적용합니다.
    """
    return get_llm_client() if llm_client is USE_DEFAULT_LLM else llm_client

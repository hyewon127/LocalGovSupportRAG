# [WBS 6.1] 서버 전체가 공유하는 자원(OpenSearch/LLM 클라이언트, 필터 후보값)과 그걸 꺼내 쓰는 의존성 함수
#
# 왜 요청마다 새로 만들지 않는가:
#   pipeline.answer_question()은 os_client/llm_client/candidates를 안 넘기면 매번 새로 만듭니다.
#   스크립트로 한 번 돌릴 때는 괜찮지만, 서버에서 요청마다 그러면
#     - candidates: 질문 1개당 OpenSearch 집계 쿼리가 1번씩 더 나감 (pipeline.py docstring에서 권장한 대로
#                   서버 시작 시 한 번만 구해서 재사용)
#     - os_client : 요청마다 HTTP 연결을 새로 맺음 (클라이언트 객체가 연결 풀을 들고 있어서 재사용이 유리)
#   그래서 main.py의 lifespan(서버 시작 시 1회 실행)에서 한 번 만들어 app.state에 두고, 각 엔드포인트는
#   아래 get_resources()로 꺼내 씁니다.
#
# 왜 routers에서 main.py를 직접 import하지 않고 이 파일을 거치는가:
#   main.py가 routers를 import하는데 routers가 다시 main.py를 import하면 순환 import가 됩니다.
#   공유 자원의 "형태"를 이 파일에 따로 두면 main.py(만드는 쪽)와 routers(쓰는 쪽)가 서로를 몰라도 됩니다.

from dataclasses import dataclass

from fastapi import Request
from openai import OpenAI
from opensearchpy import OpenSearch


@dataclass
class AppResources:
    os_client: OpenSearch
    llm_client: OpenAI | None          # .env에 키가 없으면 None (llm_client.get_llm_client() 참고)
    candidates: dict[str, list[str]]   # {"region": [...], "category": [...]} - query_slots.fetch_candidate_values()


def get_resources(request: Request) -> AppResources:
    """
    FastAPI 의존성(Depends)으로 쓰는 함수. 엔드포인트 인자에 `res: AppResources = Depends(get_resources)`라고
    적으면 FastAPI가 요청마다 이 함수를 불러서 값을 넣어줍니다.

    전역 변수 대신 이 방식을 쓰는 이유: 테스트할 때 app.dependency_overrides[get_resources]에 가짜 자원
    (가짜 LLM 등)을 끼워 넣을 수 있습니다. WBS 5에서 가짜 LLM 클라이언트로 ok/ungrounded 경로를 검증했던
    것처럼, API 단에서도 같은 방식으로 실제 키 없이 경로별 테스트가 가능해집니다.
    """
    return request.app.state.resources

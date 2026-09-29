# [WBS 6.1] src/api 패키지 초기화 - src/rag, src/indexing 모듈을 import할 수 있게 경로를 잡아둡니다.
#
# 왜 여기서 하는가:
#   src/rag/*.py와 src/indexing/*.py는 전부 "from config import ...", "from hybrid_search import ..."처럼
#   같은 폴더 기준(flat) import를 씁니다 (WBS 4·5에서 각 파일을 스크립트로 직접 실행하며 만들었기 때문).
#   FastAPI 서버는 프로젝트 루트에서 "uvicorn src.api.main:app"으로 뜨므로, 그대로는 이 import들이 실패합니다.
#
#   패키지의 __init__.py는 "src.api 안의 어떤 모듈을 import하든 가장 먼저 한 번" 실행됩니다.
#   그래서 여기서 경로를 잡아두면 main.py/routers/*.py 어디서 pipeline을 import하든 순서를 신경 쓸 필요가 없습니다.
#   (각 파일마다 sys.path 코드를 복붙하면 한 군데를 빠뜨렸을 때만 터지는 버그가 생깁니다.)
#
#   경로는 cwd가 아니라 __file__ 기준으로 계산합니다 - fetch_bizinfo.py에서 겪은 "실행 위치에 따라 경로가
#   깨지는" 버그를 반복하지 않기 위해서입니다 (src/rag/query_slots.py 상단 주석과 같은 이유).

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent  # src/api/ -> src/

for _module_dir in (_SRC_DIR / "rag", _SRC_DIR / "indexing"):
    if str(_module_dir) not in sys.path:
        sys.path.insert(0, str(_module_dir))

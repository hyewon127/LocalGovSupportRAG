# 챗봇 한 번에 실행하기 (WBS 9 데모용)
#
# 순서: OpenSearch -> (로컬 LLM 준비) -> FastAPI 백엔드(챗봇 웹 화면 포함) -> 브라우저 열기
# 챗봇 화면은 백엔드가 http://127.0.0.1:8000/ 에서 같이 서빙합니다(web/). 예전 Streamlit 화면(ui/app.py)은
# 관리·디버그용으로 남겨뒀고, -Streamlit을 붙이면 같이 띄웁니다:  scripts\start_chatbot.ps1 -Streamlit
# 이미 떠 있는 서비스는 건너뜁니다. 그래서 여러 번 실행해도 서버가 중복으로 뜨지 않습니다.
# 끌 때는 scripts\stop_chatbot.bat (또는 stop_chatbot.ps1)을 실행하세요.
#
# OpenSearch 설치 위치는 PC마다 다르므로 환경변수 OPENSEARCH_HOME으로 바꿀 수 있습니다.
# (기본값은 개발 PC의 zip 설치 경로)
# 이 파일은 UTF-8 BOM으로 저장해야 합니다 - Windows PowerShell 5.1은 BOM이 없으면 한글을 ANSI(cp949)로 읽어 깨뜨림.

param([switch]$Streamlit)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$OpenSearchHome = if ($env:OPENSEARCH_HOME) { $env:OPENSEARCH_HOME } else { "C:\Users\SMT20\opensearch-3.8.0" }
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

# localhost가 아니라 127.0.0.1 - Windows에서 localhost는 IPv6(::1)를 먼저 시도해 요청마다 약 2초 늦어짐 (README 4.3)
function Test-Url([string]$url) {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $url | Out-Null; return $true } catch { return $false }
}

function Wait-Until([string]$name, [scriptblock]$check, [int]$timeoutSec) {
    Write-Host -NoNewline "  $name 준비 대기 중"
    for ($i = 0; $i -lt $timeoutSec; $i += 2) {
        if (& $check) { Write-Host " -> 완료"; return }
        Write-Host -NoNewline "."
        Start-Sleep 2
    }
    Write-Host ""
    throw "$name 이(가) ${timeoutSec}초 안에 준비되지 않았습니다. logs 폴더의 로그를 확인하세요."
}

# 파이썬 서버를 창 없이 띄우고 출력은 logs\에 남김 (문제가 생기면 여기부터 확인)
# 인자가 길어서 splatting(@{...})으로 넘김 - 줄 끝 백틱(`) 연결은 줄바꿈 문자에 따라 깨질 수 있음
function Start-PythonServer([string]$logName, [string[]]$arguments) {
    $params = @{
        FilePath               = $Python
        ArgumentList           = $arguments
        WorkingDirectory       = $ProjectRoot
        WindowStyle            = "Hidden"
        RedirectStandardOutput = Join-Path $LogDir "$logName.out.log"
        RedirectStandardError  = Join-Path $LogDir "$logName.log"  # uvicorn/streamlit 로그는 stderr로 나옴
    }
    Start-Process @params
}

if (-not (Test-Path $Python)) { throw ".venv가 없습니다. README 4.1대로 가상환경부터 만드세요." }

# 1) OpenSearch - 켜진 직후에는 샤드 복구 중이라 status가 red입니다. red가 풀려야 검색이 됩니다.
Write-Host "[1/3] OpenSearch"
if (Test-Url "http://127.0.0.1:9200") {
    Write-Host "  이미 실행 중"
} else {
    $bat = Join-Path $OpenSearchHome "bin\opensearch.bat"
    if (-not (Test-Path $bat)) { throw "OpenSearch를 찾을 수 없습니다: $bat (환경변수 OPENSEARCH_HOME을 설정하세요)" }
    Start-Process -FilePath $bat -WindowStyle Hidden
}
Wait-Until "OpenSearch" {
    try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:9200/_cluster/health").Content -notmatch '"status":"red"' }
    catch { $false }
} 180

# 1.5) 로컬 LLM(Ollama) - 있으면 쓰고, 없어도 챗봇은 "공고 목록만 안내"로 동작하므로 실패해도 멈추지 않음.
#   qwen2.5-rag:7b는 한 번에 읽는 양을 늘린 버전(ollama/Modelfile 주석). 원본만 받아둔 경우 여기서 만들어 줌
#   (가중치 공유라 몇 초면 끝남). 백엔드가 뜰 때 모델을 고르므로 반드시 백엔드보다 먼저 해야 함.
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $models = (ollama list) -join "`n"
    if ($models -notmatch "qwen2\.5-rag:7b" -and $models -match "qwen2\.5:7b") {
        Write-Host "[LLM] qwen2.5-rag:7b 만드는 중 (ollama/Modelfile)"
        ollama create qwen2.5-rag:7b -f (Join-Path $ProjectRoot "ollama\Modelfile") | Out-Null
    } elseif ($models -notmatch "qwen2\.5") {
        Write-Host "[LLM] Ollama 모델이 없습니다. 답변 생성을 쓰려면: ollama pull qwen2.5:7b"
    }
}

# 2) 백엔드 - 임베딩 모델을 미리 올려두느라 준비까지 약 20초
Write-Host "[2/3] 백엔드 (FastAPI :8000)"
if (Test-Url "http://127.0.0.1:8000/health") {
    Write-Host "  이미 실행 중"
} else {
    Start-PythonServer "backend" @("-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000")
    Wait-Until "백엔드" { Test-Url "http://127.0.0.1:8000/health" } 120
}

# 3) Streamlit 화면 (선택)
if (-not $Streamlit) {
    Write-Host "[3/3] Streamlit 화면 - 건너뜀 (필요하면 -Streamlit)"
} elseif (Test-Url "http://127.0.0.1:8501/_stcore/health") {
    Write-Host "[3/3] Streamlit 화면 (:8501)"
    Write-Host "  이미 실행 중"
} else {
    Write-Host "[3/3] Streamlit 화면 (:8501)"
    Start-PythonServer "ui" @("-m", "streamlit", "run", "ui/app.py", "--server.headless", "true", "--server.port", "8501")
    Wait-Until "화면" { Test-Url "http://127.0.0.1:8501/_stcore/health" } 60
}

# LLM 키가 없으면 답변 문장 대신 관련 공고 목록만 나옵니다(status = llm_unavailable). 미리 알려줘야 "고장"으로 오해하지 않음.
$health = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/health").Content | ConvertFrom-Json
Write-Host ""
Write-Host "준비 완료: 공고 청크 $($health.index_docs)개 색인됨"
if ($health.llm_enabled) {
    Write-Host "LLM: 사용 가능 ($($health.llm_provider) / $($health.llm_model))"
} else {
    Write-Host "LLM: 비활성 - 답변 문장 없이 관련 공고 목록만 안내합니다. (Ollama 모델 또는 .env의 API 키 필요)"
}
Write-Host "챗봇: http://127.0.0.1:8000   API 문서: http://127.0.0.1:8000/docs"
if ($Streamlit) { Write-Host "Streamlit(관리용): http://127.0.0.1:8501" }
Start-Process "http://127.0.0.1:8000"

# 챗봇 한 번에 실행하기 (WBS 9 데모용)
#
# 순서: OpenSearch -> FastAPI 백엔드 -> Streamlit 화면 -> 브라우저 열기
# 이미 떠 있는 서비스는 건너뜁니다. 그래서 여러 번 실행해도 서버가 중복으로 뜨지 않습니다.
# 끌 때는 scripts\stop_chatbot.bat (또는 stop_chatbot.ps1)을 실행하세요.
#
# OpenSearch 설치 위치는 PC마다 다르므로 환경변수 OPENSEARCH_HOME으로 바꿀 수 있습니다.
# (기본값은 개발 PC의 zip 설치 경로)
# 이 파일은 UTF-8 BOM으로 저장해야 합니다 - Windows PowerShell 5.1은 BOM이 없으면 한글을 ANSI(cp949)로 읽어 깨뜨림.

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

# 2) 백엔드 - 임베딩 모델을 미리 올려두느라 준비까지 약 20초
Write-Host "[2/3] 백엔드 (FastAPI :8000)"
if (Test-Url "http://127.0.0.1:8000/health") {
    Write-Host "  이미 실행 중"
} else {
    Start-PythonServer "backend" @("-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000")
    Wait-Until "백엔드" { Test-Url "http://127.0.0.1:8000/health" } 120
}

# 3) 화면
Write-Host "[3/3] 화면 (Streamlit :8501)"
if (Test-Url "http://127.0.0.1:8501/_stcore/health") {
    Write-Host "  이미 실행 중"
} else {
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
    Write-Host "LLM: 비활성 - .env에 UPSTAGE_API_KEY가 없어서 답변 문장 없이 관련 공고 목록만 안내합니다."
}
Write-Host "화면: http://127.0.0.1:8501   API 문서: http://127.0.0.1:8000/docs"
Start-Process "http://127.0.0.1:8501"

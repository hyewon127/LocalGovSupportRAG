# 챗봇 끄기 - start_chatbot.ps1이 띄운 백엔드(:8000)와 화면(:8501)을 종료합니다.
# OpenSearch는 다른 작업(색인·평가)에도 쓰이므로 기본으로는 끄지 않고, -All을 붙이면 같이 끕니다.
param([switch]$All)

$ports = @(8000, 8501)
if ($All) { $ports += 9200 }

foreach ($port in $ports) {
    # 포트를 잡고 있는 프로세스를 찾아서 끔 (창을 숨겨서 띄웠기 때문에 창을 닫는 방식으로는 끌 수 없음)
    $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    if (-not $pids) { Write-Host ":$port 실행 중 아님"; continue }
    foreach ($p in $pids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Write-Host ":$port 종료 (PID $p)"
    }
}

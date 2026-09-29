# 기업마당 자동 갱신을 Windows 작업 스케줄러에 등록 (리눅스의 crontab과 같은 역할)
#
#   scripts\register_sync_task.ps1                 # 매일 06:00에 실행하도록 등록
#   scripts\register_sync_task.ps1 -Time 21:30     # 시각 바꾸기
#   scripts\register_sync_task.ps1 -Unregister     # 등록 해제
#   schtasks /Run /TN "LocalGovSupportRAG\DailySync"   # 지금 바로 한 번 실행해 보기
#
# 리눅스 서버라면 crontab -e 에 한 줄 (매일 06:00):
#   0 6 * * * cd /path/to/LocalGovSupportRAG && .venv/bin/python src/sync/sync_bizinfo.py >> logs/sync.log 2>&1
#
# 왜 하루 1번인가: 기업마당 공고는 하루에 수십 건 단위로 올라오고 접수 기간이 보통 2주 이상이라, 몇 시간 늦게 반영돼도
#   사용자가 신청 기회를 놓치지 않습니다. 새 공고가 없으면 목록 조회(몇 초)만 하고 끝나서 자주 돌려도 부담은 적지만,
#   공공 API에 불필요한 호출을 줄이는 쪽을 택했습니다.
# 전제: 실행 시각에 PC가 켜져 있고 OpenSearch가 떠 있어야 합니다. 꺼져 있으면 실패로 기록되고(data/sync_state.json),
#   다음 날 실행에서 밀린 새 공고까지 한꺼번에 처리됩니다 (기준이 "마지막 실행 이후"가 아니라 "아직 없는 공고"라서).
# 관리자 권한이 필요 없도록 현재 사용자 계정으로, 로그인해 있을 때만 실행되게 등록합니다.

param(
    [string]$Time = "06:00",
    [switch]$Unregister
)

$TaskName = "LocalGovSupportRAG\DailySync"
$Runner = Join-Path $PSScriptRoot "run_sync.bat"

if ($Unregister) {
    schtasks /Delete /TN $TaskName /F
    exit $LASTEXITCODE
}
if (-not (Test-Path $Runner)) { throw "실행 파일이 없습니다: $Runner" }

schtasks /Create /TN $TaskName /TR "`"$Runner`"" /SC DAILY /ST $Time /F | Out-Host
if ($LASTEXITCODE -ne 0) { throw "등록 실패 (종료 코드 $LASTEXITCODE)" }
Write-Host ""
Write-Host "등록 완료: 매일 $Time 에 기업마당 새 공고를 반영합니다."
Write-Host "  지금 한 번 실행: schtasks /Run /TN `"$TaskName`""
Write-Host "  상태 확인     : schtasks /Query /TN `"$TaskName`" /V /FO LIST"
Write-Host "  실행 기록     : logs\sync.log, data\sync_state.json, http://127.0.0.1:8000/sync/status"

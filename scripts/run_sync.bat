@echo off
rem Daily data sync (Windows Task Scheduler runs this). Logs: logs\sync.log, state: data\sync_state.json
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" "src\sync\sync_bizinfo.py" %*

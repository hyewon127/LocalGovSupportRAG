@echo off
rem Double-click: register daily 06:00 sync. Add -Unregister to remove.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0register_sync_task.ps1" %*
pause

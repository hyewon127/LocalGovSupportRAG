@echo off
rem Add -All to also stop OpenSearch.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_chatbot.ps1" %*
pause

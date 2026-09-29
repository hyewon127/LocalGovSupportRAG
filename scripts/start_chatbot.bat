@echo off
rem Double-click launcher. Bypasses the PowerShell execution policy for this run only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_chatbot.ps1"
pause

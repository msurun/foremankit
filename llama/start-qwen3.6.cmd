@echo off
rem Double-click launcher (bypasses the PowerShell execution policy for this one script).
rem Pass options through, e.g.:  start-qwen3.6.cmd -NoThink
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-qwen3.6.ps1" %*

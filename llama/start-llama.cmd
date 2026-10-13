@echo off
rem Double-click launcher (bypasses the PowerShell execution policy for this one script).
rem Pass options through, e.g.:  start-llama.cmd -NoThink
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-llama.ps1" %*

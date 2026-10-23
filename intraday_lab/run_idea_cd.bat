@echo off
title Intraday Lab - ideas C and D (research only - does not touch Signal Desk)
cd /d "%~dp0.."
set PY=uv run python
if exist ".venv\Scripts\python.exe" set PY=".venv\Scripts\python.exe"
%PY% -m intraday_lab cd
echo.
echo Results are in intraday_lab\results\idea_cd.txt
pause

@echo off
title Intraday Lab - idea A (research only - does not touch Signal Desk)
cd /d "%~dp0.."
set PY=uv run python
if exist ".venv\Scripts\python.exe" set PY=".venv\Scripts\python.exe"
%PY% -m intraday_lab news
if errorlevel 1 goto end
%PY% -m intraday_lab a
:end
echo.
echo Results are in intraday_lab\results\idea_a.txt
pause

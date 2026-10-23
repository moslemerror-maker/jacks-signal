@echo off
title Intraday Lab (research only - does not touch Signal Desk)
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m intraday_lab %*
) else (
  uv run python -m intraday_lab %*
)
echo.
echo Results are in intraday_lab\results\idea_b.txt
pause

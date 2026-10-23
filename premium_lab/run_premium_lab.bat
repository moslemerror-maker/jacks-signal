@echo off
title Premium Lab (research only - does not touch Signal Desk)
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m premium_lab %*
) else (
  uv run python -m premium_lab %*
)
echo.
echo Results are in premium_lab\results\report.txt
pause

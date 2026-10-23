@echo off
title Range Lab (research only - does not touch Signal Desk)
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m range_lab %*
) else (
  uv run python -m range_lab %*
)
echo.
echo Results are in range_lab\results\report.txt
pause

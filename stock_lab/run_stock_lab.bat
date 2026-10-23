@echo off
title Stock Lab (research only - does not touch Signal Desk)
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m stock_lab all %*
) else (
  uv run python -m stock_lab all %*
)
echo.
echo Results are in stock_lab\results\gate0.txt and gate1.txt
pause

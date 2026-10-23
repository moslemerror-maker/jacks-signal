@echo off
title Expiry Lab (research only - does not touch Signal Desk)
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m expiry_lab %*
) else (
  uv run python -m expiry_lab %*
)
echo.
echo Results are in expiry_lab\results\report.txt
pause

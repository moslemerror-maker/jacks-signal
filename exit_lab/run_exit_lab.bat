@echo off
title Exit Lab (research only - does not touch Signal Desk)
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m exit_lab %*
) else (
  uv run python -m exit_lab %*
)
echo.
echo Results are in exit_lab\results\report.txt
pause

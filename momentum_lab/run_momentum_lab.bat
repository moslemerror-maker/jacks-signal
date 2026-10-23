@echo off
title Momentum Lab (research only - does not touch Signal Desk)
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m momentum_lab %*
) else (
  uv run python -m momentum_lab %*
)
echo.
echo Results are in momentum_lab\results\report.txt
pause

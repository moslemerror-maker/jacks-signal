@echo off
title Signal Desk check
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m signal_desk check %*
) else (
  uv run python -m signal_desk check %*
)
pause

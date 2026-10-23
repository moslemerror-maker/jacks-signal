@echo off
title Signal Desk
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m signal_desk %*
) else (
  uv run python -m signal_desk %*
)
pause

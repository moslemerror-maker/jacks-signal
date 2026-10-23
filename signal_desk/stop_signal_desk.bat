@echo off
curl -s -X POST http://127.0.0.1:8765/api/shutdown
echo.
echo Signal Desk stopped (if it was running).
timeout /t 3 >nul

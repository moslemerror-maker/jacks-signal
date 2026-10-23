@echo off
title Remove Signal Desk autostart
powershell -NoProfile -Command "Unregister-ScheduledTask -TaskName 'Signal Desk' -Confirm:$false"
echo Autostart removed.
pause

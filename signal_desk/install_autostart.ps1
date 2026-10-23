$vbs = Join-Path $PSScriptRoot "run_hidden.vbs"
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`"" -WorkingDirectory (Split-Path $PSScriptRoot -Parent)
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 09:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 8) -MultipleInstances IgnoreNew
$desc = "Signal Desk: Nifty alert assistant (alerts only, never places orders)"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator)
$mode = ""
if ($isAdmin) {
    try {
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
        Register-ScheduledTask -TaskName "Signal Desk" -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
            -Description $desc -Force -ErrorAction Stop | Out-Null
        $mode = "full"
    } catch {
        Write-Host "Could not register the locked-screen version: $($_.Exception.Message)"
    }
}
if (-not $mode) {
    Register-ScheduledTask -TaskName "Signal Desk" -Action $action -Trigger $trigger -Settings $settings -Description $desc -Force | Out-Null
    $mode = "basic"
}

$url = Join-Path ([Environment]::GetFolderPath("Desktop")) "Signal Desk.url"
Set-Content -Path $url -Value "[InternetShortcut]`r`nURL=http://127.0.0.1:8765/" -Encoding ASCII

Write-Host ""
if ($mode -eq "full") {
    Write-Host "Done (FULL): Signal Desk starts at 09:00 every weekday even if the PC is locked or not signed in,"
    Write-Host "and wakes the PC from sleep for it. It stops itself at 15:40."
    Write-Host "Note: Windows must allow wake timers (Power Options > Sleep > Allow wake timers = Enable)."
} else {
    Write-Host "Done (BASIC): Signal Desk starts at 09:00 on weekdays while you are signed in (locked is fine, signed-out is not)."
    Write-Host "For the locked/signed-out version, right-click install_autostart.bat > Run as administrator."
}
Write-Host "Open it any time with the 'Signal Desk' shortcut on your desktop (http://127.0.0.1:8765)."

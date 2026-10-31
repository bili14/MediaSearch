@echo off
rem ============================================================
rem  MediaSearch service - STOP
rem  This file MUST stay pure ASCII (GBK cmd eats UTF-8 bytes).
rem ============================================================
setlocal
set "STOPPED="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:"127.0.0.1:8686 .*LISTENING"') do (
  taskkill /F /PID %%p >nul 2>&1 && set "STOPPED=1"
)
if defined STOPPED (
  echo [MediaSearch] service stopped (port 8686 released).
) else (
  echo [MediaSearch] no service found on port 8686.
)
endlocal

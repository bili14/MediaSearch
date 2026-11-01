@echo off
rem ============================================================
rem  MediaSearch service - START
rem  This file MUST stay pure ASCII (GBK cmd eats UTF-8 bytes).
rem  Chinese notes: README.md
rem
rem  All-in-one: one process on 127.0.0.1:8686 with four faces:
rem    web UI (/) + REST API (/search /status ...) + MCP (/mcp, /healthz)
rem    + OpenAI-style /v1/embeddings
rem  1) kill any stale instance on 127.0.0.1:8686
rem  2) start the service (repo venv preferred, else python on PATH)
rem  3) open the web UI in the default browser
rem ============================================================
setlocal
rem keep all runtime state on the public drive (model + library + cache)
set "WEMM_HOME=%~dp0"
set "WEMM_HF_HOME=%~dp0hf"
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\medisearch.exe" set "EXE=%~dp0.venv\Scripts\medisearch.exe"

rem kill anything still listening on 127.0.0.1:8686 (old service window)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:"127.0.0.1:8686 .*LISTENING"') do (
  taskkill /F /PID %%p >nul 2>&1
)

if defined EXE (
  start "MediaSearch service" "%EXE%" serve
) else (
  start "MediaSearch service" "%PY%" -m medisearch serve
)

echo [MediaSearch] service starting on http://127.0.0.1:8686 ...
echo [MediaSearch] first search warms up the model (~10-60s; weights auto-download on very first run).
timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:8686/
endlocal

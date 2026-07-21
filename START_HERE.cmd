@echo off
setlocal
cd /d "%~dp0"
start "Japanese Stock Strategy App" "%ComSpec%" /k call "%~dp0RUN_APP_INNER.cmd"
exit /b 0

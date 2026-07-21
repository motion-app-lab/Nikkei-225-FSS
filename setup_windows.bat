@echo off
setlocal
cd /d "%~dp0"
start "Stock App Setup" "%ComSpec%" /k call "%~dp0RUN_APP_INNER.cmd" setup
exit /b 0

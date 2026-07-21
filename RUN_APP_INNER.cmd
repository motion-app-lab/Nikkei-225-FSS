@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Japanese Stock Strategy App
set "LOG_FILE=%~dp0startup_error.log"
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "SETUP_ONLY="
if /I "%~1"=="setup" set "SETUP_ONLY=1"

> "%LOG_FILE%" echo Startup log - %date% %time%

echo ============================================================
echo Japanese Stock Strategy App
echo ============================================================
echo.

if not exist "%~dp0app.py" goto :not_extracted
if not exist "%~dp0requirements.txt" goto :not_extracted

echo [1/5] Detecting Python 3.13.x...
set "BASE_PY="
call :try_path_python
if defined BASE_PY goto :python_found
call :try_local_python
if defined BASE_PY goto :python_found
call :try_python_launcher
if not defined BASE_PY goto :python_missing

:python_found
echo Python version:
"%BASE_PY%" --version
echo Python executable:
"%BASE_PY%" -c "import sys; print(sys.executable)"
>> "%LOG_FILE%" "%BASE_PY%" --version 2>&1
>> "%LOG_FILE%" "%BASE_PY%" -c "import sys; print(sys.executable)" 2>&1
echo.

echo [2/5] Checking the local virtual environment...
if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
  if not errorlevel 1 goto :venv_ready
  echo Existing .venv uses a different Python version and will be recreated.
  >> "%LOG_FILE%" echo Existing .venv uses a different Python version and will be recreated.
  goto :recreate_venv
)
if exist "%VENV_DIR%" (
  echo Existing .venv is incomplete and will be recreated.
  >> "%LOG_FILE%" echo Existing .venv is incomplete and will be recreated.
  goto :recreate_venv
)
goto :create_venv

:recreate_venv
rmdir /s /q "%VENV_DIR%"
if exist "%VENV_DIR%" goto :venv_remove_error

:create_venv
echo Creating .venv with Python 3.13.x...
"%BASE_PY%" -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_error

:venv_ready
if not exist "%VENV_PY%" goto :venv_error
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
if errorlevel 1 goto :venv_error
echo Virtual environment version:
"%VENV_PY%" --version
echo Virtual environment executable:
"%VENV_PY%" -c "import sys; print(sys.executable)"
>> "%LOG_FILE%" "%VENV_PY%" --version 2>&1
>> "%LOG_FILE%" "%VENV_PY%" -c "import sys; print(sys.executable)" 2>&1
echo.

echo [3/5] Checking required libraries and the saved Nikkei model environment...
"%VENV_PY%" -c "import fastapi,uvicorn,jinja2,dotenv,pandas,numpy,yfinance,fredapi,sklearn,joblib,catboost,matplotlib,httpx,pytest,holidays; from services.nikkei_artifact import load_manifest,_validate_runtime; _validate_runtime(load_manifest())" >nul 2>&1
if not errorlevel 1 goto :packages_ready

echo [4/5] Installing required libraries. This may take several minutes...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :pip_error
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :pip_error

"%VENV_PY%" -c "import fastapi,uvicorn,jinja2,dotenv,pandas,numpy,yfinance,fredapi,sklearn,joblib,catboost,matplotlib,httpx,pytest,holidays; from services.nikkei_artifact import load_manifest,_validate_runtime; _validate_runtime(load_manifest())"
if errorlevel 1 goto :package_import_error

:packages_ready
echo.
echo Setup check completed successfully.
>> "%LOG_FILE%" echo Setup check completed successfully.
if defined SETUP_ONLY exit /b 0

set "PROJECT_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 } catch { exit 1 }; if(-not $r.project_root){exit 2}; $expected=[IO.Path]::GetFullPath($env:PROJECT_ROOT).TrimEnd([char]92); $actual=[IO.Path]::GetFullPath([string]$r.project_root).TrimEnd([char]92); if($r.status -eq 'ok' -and [string]::Equals($actual,$expected,[StringComparison]::OrdinalIgnoreCase)){exit 0}; exit 2" >nul 2>&1
if "%ERRORLEVEL%"=="0" goto :already_running
if "%ERRORLEVEL%"=="2" goto :port_conflict

echo.
echo [5/5] Starting the FastAPI app...
echo Keep this window open while using the app.
echo URL: http://127.0.0.1:8000
echo.
set "STOCK_APP_OPEN_BROWSER=1"
set "APP_ENTRY=%~dp0app.py"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$command=[char]34+$env:VENV_PY+[char]34+' '+[char]34+$env:APP_ENTRY+[char]34+' 2>&1'; cmd.exe /d /s /c $command | ForEach-Object { Write-Host $_; [System.IO.File]::AppendAllText($env:LOG_FILE, [string]$_+[Environment]::NewLine, [System.Text.UTF8Encoding]::new($false)) }; $appExit=$LASTEXITCODE; exit $appExit"
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" goto :app_error

echo.
echo The app has stopped.
exit /b 0

:already_running
echo This project is already running. Opening the browser...
start "" "http://127.0.0.1:8000"
exit /b 0

:port_conflict
call :write_error "Port 8000 is being used by a different or unknown app. Close that app manually, then run START_HERE.cmd again. No process was stopped automatically."
goto :fatal

:try_path_python
where python >nul 2>&1
if errorlevel 1 exit /b 0
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "BASE_PY=%%P"
exit /b 0

:try_local_python
set "LOCAL_PY=%LocalAppData%\Programs\Python\Python313\python.exe"
if not exist "%LOCAL_PY%" exit /b 0
"%LOCAL_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
set "BASE_PY=%LOCAL_PY%"
exit /b 0

:try_python_launcher
where py >nul 2>&1
if errorlevel 1 exit /b 0
py -3.13 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
for /f "delims=" %%P in ('py -3.13 -c "import sys; print(sys.executable)"') do set "BASE_PY=%%P"
exit /b 0

:not_extracted
call :write_error "The app files were not found. Extract the ZIP first, then run START_HERE.cmd from the extracted folder."
goto :fatal

:python_missing
call :write_error "Python 3.13.x was not found. Install a Python 3.13 release, enable the python command, then run START_HERE.cmd again."
goto :fatal

:venv_remove_error
call :write_error "The old .venv folder could not be removed. Close any app processes using it, then run START_HERE.cmd again."
goto :fatal

:venv_error
call :write_error "The local Python 3.13.x environment could not be created. Close any process using .venv, then try again."
goto :fatal

:pip_error
call :write_error "Required libraries could not be installed. Check the internet connection and free disk space, then try again."
goto :fatal

:package_import_error
call :write_error "Required libraries were installed but could not be loaded with Python 3.13.x. Review the log and rebuild .venv."
goto :fatal

:app_error
call :write_error "The app stopped with an error. Review the messages shown above and startup_error.log."
goto :fatal

:write_error
echo ERROR: %~1
>> "%LOG_FILE%" echo ERROR: %~1
exit /b 0

:fatal
echo.
echo Startup failed.
echo This window will remain open so the error can be read.
echo Log file: %LOG_FILE%
echo.
pause
exit /b 1

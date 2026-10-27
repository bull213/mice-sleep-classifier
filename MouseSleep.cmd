@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run Setup.cmd first.
    exit /b 1
)
".venv\Scripts\python.exe" -m mouse_sleep %*
exit /b %errorlevel%

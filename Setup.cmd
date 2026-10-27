@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Install 64-bit Python 3.12 with the Windows Python launcher, then run Setup.cmd again.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 (
    pause
    exit /b 1
)
echo Ready. Run Start_MouseSleep.cmd to open the desktop application.
echo Before analysis, install the segmentation models using docs\INSTALLATION.md.
echo Then download or train the classifier under Classifier setup.
pause

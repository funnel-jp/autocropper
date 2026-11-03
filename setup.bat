@echo off
cd /d %~dp0
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 or later
    echo         and make sure "Add python.exe to PATH" is checked.
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
python setup.py %*
echo.
pause

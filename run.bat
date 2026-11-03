@echo off
cd /d %~dp0
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup.bat first to set up the environment.
    pause
    exit /b 1
)
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    start "Ollama Server" /MIN "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
) else (
    where ollama >nul 2>nul && start "Ollama Server" /MIN ollama serve
)
".venv\Scripts\python.exe" app.py

pause

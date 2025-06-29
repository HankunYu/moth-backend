@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo           Moth System Startup
echo ===========================================
echo.

:: Step 1: Update IP configurations
echo [1/3] Updating IP configurations...
call check_ip.bat
if errorlevel 1 (
    echo Failed to update IP configurations
    pause
    exit /b 1
)

echo.
echo IP configurations updated successfully!
echo Waiting 1 second before starting Unity game...
timeout /t 1 /nobreak >nul

:: Step 2: Start Unity game
echo.
echo [2/3] Starting Unity game...
set "UNITY_EXE_PATH=C:\path\to\your\unity\game.exe"
echo Starting Unity game at: !UNITY_EXE_PATH!
if exist "!UNITY_EXE_PATH!" (
    start "" "!UNITY_EXE_PATH!"
    echo Unity game started successfully
) else (
    echo ERROR: Unity executable not found at !UNITY_EXE_PATH!
    echo Please update the UNITY_EXE_PATH in this script
    pause
    exit /b 1
)

echo Waiting 2 seconds before starting Python controller...
timeout /t 2 /nobreak >nul

:: Step 3: Start Python controller in WSL
echo.
echo [3/3] Starting Python moth controller in WSL...
set "WSL_PROJECT_DIR=/mnt/e/GitHub/moth-backend"
echo Switching to WSL directory: !WSL_PROJECT_DIR!

:: Start Python controller in WSL with virtual environment
echo Activating virtual environment and running moth_controller.py...
wsl -d Ubuntu -e bash -c "cd !WSL_PROJECT_DIR! && source venv1/bin/activate && python3 moth_controller.py"

echo.
echo Moth system startup completed!
pause
@echo off
echo Starting script...
pause

setlocal enabledelayedexpansion
echo Variables enabled
pause

echo Current directory: %CD%
pause

echo Checking if config.json exists...
if exist "config.json" (
    echo config.json found
) else (
    echo config.json NOT found
)
pause

echo Testing ipconfig command...
ipconfig | findstr /i "IPv4"
pause

echo Testing WSL command...
wsl --version
pause

echo Testing PowerShell...
powershell -Command "Write-Host 'PowerShell test successful'"
pause

echo Script completed
pause
@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo        IP Detection and Config Update
echo ===========================================
echo.

:: Check if config.json exists
if not exist "config.json" (
    echo ERROR: config.json not found in current directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Detecting IP addresses...
echo.

:: Get the main network adapter IP (skip 127.0.0.1)
set "host_ip="
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /i "IPv4"') do (
    set "ip=%%i"
    set "ip=!ip: =!"
    if not "!ip!"=="127.0.0.1" if not "!ip!"=="" (
        if "!host_ip!"=="" set "host_ip=!ip!"
        echo Found Windows IP: !ip!
    )
)

if "!host_ip!"=="" (
    echo ERROR: No valid Windows IP address found
    echo Available IPs:
    ipconfig | findstr /i "IPv4"
    pause
    exit /b 1
)

:: Get WSL IP
echo.
echo Getting WSL IP...
for /f %%i in ('wsl hostname -I 2^>nul') do (
    set "wsl_ip=%%i"
    echo Found WSL IP: !wsl_ip!
)

if "!wsl_ip!"=="" (
    echo WSL not found or not running
    pause
    exit /b 1
)

echo.
echo Current IPs:
echo Windows Host IP: !host_ip!
echo WSL IP: !wsl_ip!
echo.

:: Update config.json using PowerShell
echo Updating config.json...
echo Executing PowerShell command...
powershell.exe -ExecutionPolicy Bypass -Command "& {try {$config = Get-Content 'config.json' | ConvertFrom-Json; $config.udp_client.host = '!host_ip!'; $config.ollama.host = 'http://!host_ip!:11434'; $config | ConvertTo-Json -Depth 10 | Set-Content 'config.json'; Write-Host 'Config.json updated successfully'} catch {Write-Host 'Error updating config.json:' $_.Exception.Message; exit 1}}"

if errorlevel 1 (
    echo Failed to update config.json
    pause
    exit /b 1
)

echo Successfully updated config.json with Windows Host IP: !host_ip!
echo UDP Client Host: !host_ip!
echo Ollama Host: http://!host_ip!:11434

:: Update Unity config file
echo.
echo Updating Unity config file...
powershell.exe -ExecutionPolicy Bypass -Command "& {try {$config = Get-Content 'config.json' | ConvertFrom-Json; $unityConfigPath = $config.unity.config_path; Write-Host 'Unity config path:' $unityConfigPath; if (Test-Path (Split-Path $unityConfigPath)) {$unityConfig = @{'ipAddress' = '!wsl_ip!'; 'sendPort' = 8888; 'receivePort' = 8889}; $unityConfig | ConvertTo-Json | Set-Content $unityConfigPath; Write-Host 'Unity config updated successfully'} else {Write-Host 'Unity config directory does not exist:' (Split-Path $unityConfigPath)}} catch {Write-Host 'Error updating Unity config:' $_.Exception.Message}}"

echo Unity config update completed

echo.
echo Updated files:
echo config.json UDP/Ollama hosts:
type config.json | findstr "host"
echo.
echo Unity config file updated at the configured path

echo.
echo ===========================================
pause
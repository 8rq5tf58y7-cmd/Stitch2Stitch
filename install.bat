@echo off
REM Stitch2Stitch Windows Batch Installer
REM Simple batch file installer for Windows

echo ========================================
echo Stitch2Stitch Windows Installer
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: PowerShell is required but not found.
    echo Please install PowerShell or use manual installation.
    pause
    exit /b 1
)

REM Run PowerShell installer
echo Running PowerShell installer...
powershell.exe -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation completed successfully!
) else (
    echo.
    echo Installation failed. Please check the error messages above.
)

pause


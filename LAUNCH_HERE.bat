@echo off
REM Double-click this file to launch Stitch2Stitch
REM This wrapper ensures the terminal stays open

REM Launch in a new CMD window that won't auto-close
start "Stitch2Stitch" cmd /k "cd /d "%~dp0" && launch_windows.bat"

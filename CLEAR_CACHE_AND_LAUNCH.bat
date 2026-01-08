@echo off
echo ============================================================
echo   CLEARING PYTHON CACHE AND RELAUNCHING
echo ============================================================
echo.

cd /d "%~dp0"

echo Clearing all Python cache files...
if exist "src\__pycache__" rmdir /s /q "src\__pycache__"
if exist "src\core\__pycache__" rmdir /s /q "src\core\__pycache__"
if exist "src\gui\__pycache__" rmdir /s /q "src\gui\__pycache__"
if exist "src\ml\__pycache__" rmdir /s /q "src\ml\__pycache__"
if exist "src\utils\__pycache__" rmdir /s /q "src\utils\__pycache__"
if exist "src\external\__pycache__" rmdir /s /q "src\external\__pycache__"

echo Cache cleared!
echo.
echo Launching application...
echo.

call launch_windows.bat

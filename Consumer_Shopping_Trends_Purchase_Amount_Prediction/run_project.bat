@echo off
cd /d "%~dp0"
python download_dataset.py
if errorlevel 1 pause & exit /b 1
python src\train.py
pause

@echo off
cd /d "%~dp0"
python -B "%~dp0ga_ui.py"
if errorlevel 1 pause

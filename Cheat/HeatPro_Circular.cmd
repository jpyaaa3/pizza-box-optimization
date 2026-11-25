@echo off
cd /d "%~dp0"
python -B "%~dp0heatpro_circular_ui.py"
if errorlevel 1 pause

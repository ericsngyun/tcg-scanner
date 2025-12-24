@echo off
REM Daily Price Update Script for TCG Scanner
REM Run this via Windows Task Scheduler at 2 PM daily

REM Navigate to project directory
cd /d "%~dp0..\.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the price update
python tools\data_collection\update_prices.py

REM Log completion
echo Price update completed at %date% %time% >> logs\scheduler.log

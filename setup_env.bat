@echo off
REM Setup script for Drivable Corridor project
REM This script sets up the Python environment and PATH for development

echo Setting up Python environment for Drivable Corridor project...

REM Add Python to PATH for this session
set PYTHON_PATH=C:\Users\aramesh\AppData\Local\Programs\Python\Python311
set PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%

REM Verify Python installation
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Python not found! Please install Python 3.11 or later.
    pause
    exit /b 1
)

echo Python setup complete!
echo.
echo Available commands:
echo   python test_setup.py          - Test all dependencies
echo   cd scripts ^&^& python train.py     - Run training
echo   cd scripts ^&^& python inference.py - Run inference
echo   pip install -r requirements.txt   - Install dependencies
echo.

REM Keep the command prompt open
cmd /k

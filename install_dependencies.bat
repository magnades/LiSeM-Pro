@echo off
echo =============================================================
echo   Installing RC Durability Framework Dependencies
echo =============================================================
echo.

:: 1. Upgrade pip first (Critical for older TF versions)
python -m pip install --upgrade pip

:: 2. Install specific TensorFlow version separately to ensure stability
echo Installing TensorFlow 2.10.1...
pip install tensorflow==2.10.1

:: 3. Install the rest of the requirements
echo Installing remaining dependencies...
pip install -r requirements.txt

echo.
echo =============================================================
echo   Installation Complete. Please run main.py to verify.
echo =============================================================
pause
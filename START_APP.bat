@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo  PlacePredict AI - Starting Flask Server
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Creating it...
    python -m venv --system-site-packages .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
)

start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 7; Start-Process 'http://127.0.0.1:5000'"

.\.venv\Scripts\python.exe app.py

echo.
echo Server stopped.
pause

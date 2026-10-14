@echo off
cd /d "%~dp0"
start "" http://localhost:8765/index.html
".venv\Scripts\python.exe" serve.py 8765

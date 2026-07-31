@echo off
cd /d "%~dp0"
call "%~dp0venv\Scripts\activate.bat"
python app.py

@echo off
echo Iniciando Profanity Filter...
cd /d "%~dp0"
call venv\Scripts\activate
python app.py
pause
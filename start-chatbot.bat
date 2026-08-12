@echo off
REM Double-click this file to start the chatbot.
REM The app will open in your browser automatically.
REM Click the "Stop & quit" button in the sidebar to close.

cd /d "%~dp0"
set STREAMLIT_SERVER_HEADLESS=true
set HF_MIRROR=1
call .venv\Scripts\activate.bat
streamlit run app.py --server.port=8520
pause

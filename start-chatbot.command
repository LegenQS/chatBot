#!/bin/bash
# Double-click this file to start the chatbot.
# The app will open in your browser automatically.
# Click the "🛑 停止并退出 / Stop & quit" button in the sidebar to close.

cd "$(dirname "$0")"
export STREAMLIT_SERVER_HEADLESS=true
export HF_MIRROR=1
source .venv/bin/activate
streamlit run app.py --server.port=8520

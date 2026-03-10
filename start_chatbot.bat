@echo off
REM Enterprise RAG Chatbot Startup Script

echo ========================================
echo Enterprise RAG Chatbot
echo ========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
    echo.
)

echo Starting chatbot...
echo.
echo The chatbot will open in your web browser.
echo Press Ctrl+C to stop the server.
echo.

REM Start Streamlit
streamlit run chatbot.py --server.port=8501 --server.headless=true

pause

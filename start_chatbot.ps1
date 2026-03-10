# Enterprise RAG Chatbot Startup Script (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Enterprise RAG Chatbot" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is installed
$streamlitInstalled = python -c "import streamlit" 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host ""
}

Write-Host "Starting chatbot..." -ForegroundColor Green
Write-Host "The chatbot will open in your web browser." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Green
Write-Host ""

# Start Streamlit
streamlit run chatbot.py --server.port=8501

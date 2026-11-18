@echo off
REM One-command start (Windows). Needs internet ONCE (installs + model download).
cd /d "%~dp0"
if not exist .venv ( echo creating .venv & py -3 -m venv .venv )
call .venv\Scripts\activate.bat
python -c "import sellbird, psutil" 2>nul || ( echo installing Sellbird, first run only & pip install -q -e . -r requirements-local.txt )
python scripts\verify_local.py embeddings >nul 2>nul || ( echo caching the embedding model once, needs internet & python scripts\verify_local.py embeddings --download >nul 2>nul )
where ollama >nul 2>nul || ( echo Ollama is not installed. Get it from https://ollama.com then run this again. & exit /b 1 )
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>nul || ( echo starting ollama serve & start /b ollama serve & timeout /t 4 >nul )
if "%SELLBIRD_LOCAL_MODEL%"=="" for /f %%m in ('python -c "from sellbird.local_sales.hardware import recommend_model as r; print(r()['recommended_model'])"') do set SELLBIRD_LOCAL_MODEL=%%m
echo model for this machine: %SELLBIRD_LOCAL_MODEL%
ollama list | findstr /b /c:"%SELLBIRD_LOCAL_MODEL% " >nul || ollama pull %SELLBIRD_LOCAL_MODEL%
echo open http://127.0.0.1:8765  (Ctrl+C to stop)
python -m sellbird.local_sales serve

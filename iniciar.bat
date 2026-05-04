@echo off

REM Ir para a pasta do projeto
cd /d C:\Users\alex1\OneDrive\Documentos\avalia

REM Ativar o ambiente virtual
call venv\Scripts\activate

REM Rodar o FastAPI com uvicorn
uvicorn main:app --reload

pause
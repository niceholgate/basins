cd "c:\dev\python_projects\basins\venv\Scripts"
./activate.ps1
cd "../.."
uvicorn src.api.controller:app --host=0.0.0.0 --port=8001 --reload
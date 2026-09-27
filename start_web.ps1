$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Environnement absent. Exécutez d'abord : .\setup.ps1"
}

Write-Host "Interface : http://127.0.0.1:5000"
& ".venv\Scripts\python.exe" app.py

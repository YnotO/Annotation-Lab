$ErrorActionPreference = 'Stop'
$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    throw 'Create the Python environment first using the instructions in README.md.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $appPython -m streamlit run app.py --server.headless true
}
finally {
    Pop-Location
}

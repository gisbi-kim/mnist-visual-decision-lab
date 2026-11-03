param([int]$TrainingProcessId)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($TrainingProcessId) { Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue }
$status = Get-Content -LiteralPath 'runs/main/status.json' -Raw | ConvertFrom-Json
if ($status.phase -ne 'trained') { throw 'Training did not finish successfully; refusing final evaluation.' }
& .\.venv\Scripts\python.exe -u evaluate.py --checkpoint runs/main/best --device cuda --calibrate --output results
if ($LASTEXITCODE -ne 0) { throw 'Final evaluation failed' }
& .\.venv\Scripts\python.exe -u diagnostics.py --checkpoint runs/main/best --device cuda --count 64 --output results/diagnostics-cuda.json
if ($LASTEXITCODE -ne 0) { throw 'CUDA diagnostics failed' }
& .\.venv\Scripts\python.exe -u diagnostics.py --checkpoint runs/main/best --device cpu --count 8 --output results/diagnostics-cpu.json
if ($LASTEXITCODE -ne 0) { throw 'CPU diagnostics failed' }
& .\.venv\Scripts\python.exe compare_platforms.py
if ($LASTEXITCODE -ne 0) { throw 'Platform comparison failed' }
& .\.venv\Scripts\python.exe make_report.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
Write-Output 'LOCAL_FINAL_EVALUATION_COMPLETE'

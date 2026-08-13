param(
  [switch]$SkipFrontend,
  [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "未找到 backend\.venv，请先按 README 安装后端依赖。" }

$env:PYTHONPATH = Join-Path $projectRoot "backend"
& $python -m compileall -q (Join-Path $projectRoot "backend\app")
if (-not $SkipTests) { & $python -m pytest (Join-Path $projectRoot "backend\tests") }

if (-not $SkipFrontend) {
  $node = "C:\Users\PY\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
  $tsc = Join-Path $projectRoot "frontend\node_modules\typescript\bin\tsc"
  if ((Test-Path $node) -and (Test-Path $tsc)) { Push-Location (Join-Path $projectRoot "frontend"); & $node $tsc -b --pretty false; Pop-Location }
  else { Write-Warning "未找到 Node/TypeScript，跳过前端检查。" }
}

Write-Host "Harness checks passed." -ForegroundColor Green

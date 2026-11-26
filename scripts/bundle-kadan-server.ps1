param(
    [string]$WorkerBin = $env:KADAN_WORKER_BIN
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $Root "apps\kadan-server\src-tauri\resources\runtime"

if (Test-Path $RuntimeDir) { Remove-Item -Recurse -Force $RuntimeDir }
New-Item -ItemType Directory -Force -Path `
    (Join-Path $RuntimeDir "services\api"),
    (Join-Path $RuntimeDir "services\tracing"),
    (Join-Path $RuntimeDir "services\tsuchi"),
    (Join-Path $RuntimeDir "services\hikari\api"),
    (Join-Path $RuntimeDir "services\kadan"),
    (Join-Path $RuntimeDir "web"),
    (Join-Path $RuntimeDir "bin") | Out-Null

Write-Host "[1/4] Building unified frontend (Kadan + Hikari + Tsuchi)..."
Push-Location (Join-Path $Root "services\frontend")
try {
    npm ci --no-audit --no-fund
    npm run build
    Copy-Item -Recurse -Force (Join-Path $Root "services\frontend\dist\*") (Join-Path $RuntimeDir "web")
} finally {
    Pop-Location
}

Write-Host "[2/4] Staging Python services..."
robocopy (Join-Path $Root "services\api") (Join-Path $RuntimeDir "services\api") /E /XD __pycache__ tests .pytest_cache | Out-Null
robocopy (Join-Path $Root "services\tracing") (Join-Path $RuntimeDir "services\tracing") /E /XD __pycache__ tests .pytest_cache | Out-Null
robocopy (Join-Path $Root "services\tsuchi") (Join-Path $RuntimeDir "services\tsuchi") /E /XD __pycache__ tests .pytest_cache | Out-Null
robocopy (Join-Path $Root "services\hikari\api") (Join-Path $RuntimeDir "services\hikari\api") /E /XD __pycache__ tests .pytest_cache | Out-Null
robocopy (Join-Path $Root "services\kadan") (Join-Path $RuntimeDir "services\kadan") /E /XD __pycache__ tests .pytest_cache | Out-Null
$global:LASTEXITCODE = 0

Write-Host "[3/4] Creating standalone Python runtime..."
$VenvDir = Join-Path $RuntimeDir "venv"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv venv --relocatable --python 3.12 $VenvDir
    uv pip install --python (Join-Path $VenvDir "Scripts\python.exe") `
        (Join-Path $RuntimeDir "services\tracing") `
        -r (Join-Path $Root "services\api\requirements.txt")
} else {
    python -m venv $VenvDir
    & (Join-Path $VenvDir "Scripts\pip.exe") install --upgrade pip
    & (Join-Path $VenvDir "Scripts\pip.exe") install `
        (Join-Path $RuntimeDir "services\tracing") `
        -r (Join-Path $Root "services\api\requirements.txt")
}

Write-Host "[4/4] Staging native C++ kadan-worker binary..."
if ($WorkerBin -and (Test-Path $WorkerBin)) {
    Copy-Item -Force $WorkerBin (Join-Path $RuntimeDir "bin\kadan-worker.exe")
    Write-Host "  Staged kadan-worker.exe from $WorkerBin"
} else {
    Write-Host "  Note: no WorkerBin provided."
}

Write-Host "Done! Staged Kadan Server runtime in $RuntimeDir"

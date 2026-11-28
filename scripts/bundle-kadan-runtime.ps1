param(
    [string]$WorkerBin = $env:KADAN_WORKER_BIN
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $Root "apps\kadan\src-tauri\resources\runtime"

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

Write-Host "[4/4] Staging native C++ kadan-worker runtime..."
if ($WorkerBin -and (Test-Path $WorkerBin)) {
    if (Test-Path $WorkerBin -PathType Container) {
        foreach ($sub in @("bin", "libexec", "lib", "share")) {
            $src = Join-Path $WorkerBin $sub
            if (Test-Path $src) {
                robocopy $src (Join-Path $RuntimeDir $sub) /E | Out-Null
            }
        }
        Write-Host "  Staged kadan-worker install prefix from $WorkerBin"
    } else {
        Copy-Item -Force $WorkerBin (Join-Path $RuntimeDir "bin\kadan-worker.exe")
        Write-Host "  Staged kadan-worker.exe from $WorkerBin"
    }
    $exe = Join-Path $RuntimeDir "bin\kadan-worker.exe"
    $caps = (& $exe --list-workloads | ConvertFrom-Json)
    $found = @($caps.workloads | ForEach-Object { $_.id })
    $missing = @("llm", "speech", "tts", "image" | Where-Object { $found -notcontains $_ })
    if ($missing.Count -gt 0) {
        throw "kadan-worker.exe is missing required workloads: $($missing -join ', ') (found: $($found -join ', '))"
    }
    Write-Host "  Verified all workloads compiled into kadan-worker.exe: $($found -join ', ')"
} elseif ($env:KADAN_ALLOW_NO_WORKER -eq "1") {
    Write-Host "  Note: no WorkerBin provided (KADAN_ALLOW_NO_WORKER=1)."
} else {
    throw "No KADAN_WORKER_BIN provided. Build kadan-worker.exe with all 4 CUDA workloads or set KADAN_ALLOW_NO_WORKER=1."
}

Write-Host "Done! Staged Kadan Server runtime in $RuntimeDir"

# =============================================
# StratML CLI Installer — PowerShell
# =============================================
# Usage (from project root or stratml/cli/):
#   powershell -ExecutionPolicy Bypass -File stratml\cli\install.ps1

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$mainPy     = Join-Path $scriptDir "main.py"
$binPath    = Join-Path $env:USERPROFILE "bin"
$wrapperDst = Join-Path $binPath "stratml.bat"

# ── Uninstall ────────────────────────────────
if ($Uninstall) {
    if (Test-Path $wrapperDst) {
        Remove-Item $wrapperDst -Force
        Write-Host "Removed $wrapperDst"
    }
    Write-Host "Uninstall complete."
    exit 0
}

# ── Verify Python ────────────────────────────
$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 12) { $python = $candidate; break }
            Write-Warning "Found $ver but Python 3.12+ is required."
        }
    } catch {}
}
if (-not $python) {
    Write-Error "Python 3.12+ not found. Install from https://python.org and re-run."
    exit 1
}
Write-Host "Using: $python ($( & $python --version 2>&1 ))"

# ── Install dependencies ──────────────────────
Write-Host ""
Write-Host "Installing dependencies from $projectRoot ..."

$uvAvailable = $null
try { $uvAvailable = (Get-Command uv -ErrorAction Stop).Source } catch {}

if ($uvAvailable) {
    Write-Host "Using uv ..."
    Push-Location $projectRoot
    & uv sync
    Pop-Location
} else {
    Write-Host "Using pip ..."
    $reqFile = Join-Path $projectRoot "requirements.txt"
    if (Test-Path $reqFile) {
        & $python -m pip install -r $reqFile --quiet
    } else {
        & $python -m pip install -e $projectRoot --quiet
    }
}

# ── Create bin dir ────────────────────────────
if (!(Test-Path $binPath)) {
    New-Item -ItemType Directory -Path $binPath | Out-Null
    Write-Host "Created $binPath"
}

# ── Write wrapper bat ─────────────────────────
@"
@echo off
"$python" "$mainPy" %*
"@ | Set-Content -Path $wrapperDst -Encoding ASCII
Write-Host "Wrote wrapper: $wrapperDst"

# ── Add bin to user PATH ──────────────────────
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$binPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$binPath;$userPath", "User")
    Write-Host "Added $binPath to user PATH"
} else {
    Write-Host "$binPath already in PATH"
}

# ── Done ──────────────────────────────────────
Write-Host ""
Write-Host "Installation complete!"
Write-Host "Restart your terminal, then verify with:"
Write-Host "  stratml doctor"

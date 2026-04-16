# =========================
# StratML CLI Installer
# =========================

$projectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$pythonExe   = "$projectRoot\.venv\Scripts\python.exe"
$mainPy      = "$projectRoot\stratml\cli\main.py"
$binPath     = "$env:USERPROFILE\bin"
$batDest     = "$binPath\stratml.bat"

Write-Host "Installing StratML CLI..."
Write-Host "  Project root : $projectRoot"
Write-Host "  Python       : $pythonExe"

if (!(Test-Path $pythonExe)) {
    Write-Host "ERROR: venv not found at $pythonExe"
    Write-Host "Run 'uv sync' or 'pip install -r requirements.txt' first."
    exit 1
}

# 1. Create bin folder if not exists
if (!(Test-Path $binPath)) {
    New-Item -ItemType Directory -Path $binPath | Out-Null
    Write-Host "Created $binPath"
}

# 2. Write bat with resolved absolute paths, explicit CRLF line endings
$batContent = "@echo off`r`n`"$pythonExe`" `"$mainPy`" %*`r`n"
[System.IO.File]::WriteAllText($batDest, $batContent, [System.Text.Encoding]::ASCII)

Write-Host "Written stratml.bat to $batDest"

# 3. Add bin to PATH if missing
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$binPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$binPath;$userPath", "User")
    Write-Host "Added $binPath to PATH"
} else {
    Write-Host "$binPath already in PATH"
}

Write-Host ""
Write-Host "Installation complete!"
Write-Host "Restart your terminal and run: stratml init"

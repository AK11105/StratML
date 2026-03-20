# =========================
# StratML CLI Installer
# =========================

$binPath = "$env:USERPROFILE\bin"
$batSource = "$PSScriptRoot\stratml.bat"
$batDest = "$binPath\stratml.bat"

Write-Host "Installing StratML CLI..."

# 1. Create bin folder if not exists
if (!(Test-Path $binPath)) {
    New-Item -ItemType Directory -Path $binPath | Out-Null
    Write-Host "Created $binPath"
}

# 2. Copy bat file
Copy-Item $batSource $batDest -Force
Write-Host "Copied stratml.bat to $binPath"

# 3. Get current user PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

# 4. Add to PATH if not already present
if ($userPath -notlike "*$binPath*") {
    $newPath = "$binPath;$userPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Added $binPath to PATH"
} else {
    Write-Host "$binPath already in PATH"
}

# 5. Done
Write-Host ""
Write-Host "Installation complete!"
Write-Host "Restart your terminal and run: stratml init"
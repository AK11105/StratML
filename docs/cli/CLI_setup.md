# StratML CLI — Setup Guide

---

## Windows

**1. Run the installer** from the `CLI/` directory in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

This copies `stratml.bat` to `%USERPROFILE%\bin` and adds it to your user `PATH`.

**2. Restart your terminal**, then confirm:

```powershell
stratml init
```

---

## Linux / macOS

**1. Make the installer executable** and run it from the `CLI/` directory:

```bash
chmod +x install.sh
./install.sh
```

This copies `stratml.sh` to `~/.local/bin/stratml`, makes it executable, and adds `~/.local/bin` to your `PATH` in `~/.bashrc` or `~/.zshrc`.

**2. Reload your shell**, then confirm:

```bash
source ~/.bashrc   # or ~/.zshrc
stratml init
```

---

## What the installers do

| Step | Windows (`install.ps1`) | Linux/macOS (`install.sh`) |
|------|------------------------|---------------------------|
| Bin directory | `%USERPROFILE%\bin` | `~/.local/bin` |
| Wrapper copied | `stratml.bat` | `stratml.sh` → `stratml` |
| PATH updated in | User environment variables | `~/.bashrc` or `~/.zshrc` |
| Requires restart | Yes | Yes (or `source` the RC file) |

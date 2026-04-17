# CLI Setup

## Requirements

- Python 3.12+
- `uv` (recommended) or `pip`

---

## Install — Linux / macOS

Run from the project root:

```bash
bash stratml/cli/install.sh
source ~/.bashrc   # or ~/.zshrc on macOS
```

The installer:
1. Verifies Python 3.12+
2. Installs dependencies via `uv sync` (falls back to `pip`)
3. Writes an executable wrapper to `~/.local/bin/stratml`
4. Adds `~/.local/bin` to your PATH in your shell RC file

---

## Install — Windows (PowerShell)

Run from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File stratml\cli\install.ps1
```

The installer:
1. Verifies Python 3.12+
2. Installs dependencies via `uv sync` (falls back to `pip`)
3. Writes `stratml.bat` to `%USERPROFILE%\bin`
4. Adds `%USERPROFILE%\bin` to your user PATH

Restart your terminal after running.

---

## Uninstall

```bash
bash stratml/cli/install.sh --uninstall        # Linux/macOS
```

```powershell
powershell -ExecutionPolicy Bypass -File stratml\cli\install.ps1 -Uninstall   # Windows
```

---

## Verify

```bash
stratml doctor
```

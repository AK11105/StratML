# CLI Setup

## Install (Windows)

Run this once from the project root:

```bash
powershell -ExecutionPolicy Bypass -File install.ps1
```

Restart your terminal or VSCode after install.

Confirm it worked:

```bash
stratml init
```

---

## Install (bash / WSL)

```bash
pip install -e .
```

Confirm:

```bash
stratml init
```

---

## Environment

Requires a `.env` file in the project root with at minimum:

```
GROQ_API_KEY=your_key_here
```

Without `GROQ_API_KEY` the system runs fully on rule-based fallbacks — all LLM paths
are skipped gracefully.

Copy `sample.env` to `.env` and fill in your key.

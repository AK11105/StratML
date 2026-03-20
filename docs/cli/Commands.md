# StratML CLI — Commands

## Init

Generate a default `config.yaml` in the current directory:

```bash
stratml init
```

---

## Validate Config

Check a config file for errors before running:

```bash
stratml validate-config <path/to/config.yaml>
```

---

## Profile Data

Inspect a dataset before running a pipeline:

```bash
stratml profile-data --input <path/to/dataset.csv>
```

---

## Doctor

Check that the environment and dependencies are correctly set up:

```bash
stratml doctor
```

---

## Run

Execute the AutoML pipeline:

```bash
stratml run --config <path/to/config.yaml>
```

**Optional flags:**

| Flag | Description |
|------|-------------|
| `--path <path>` | Override dataset path from config |
| `--mode <beginner\|intermediate\|expert>` | Override execution mode |
| `--max-iter <n>` | Cap the number of iterations |
| `--dry-run` | Print resolved config without running |

**Example:**

```bash
stratml run --config config.yaml --mode expert --max-iter 10 --dry-run
```

---

## Phase 2 (Planned)

```bash
stratml --version
stratml --help
stratml status
stratml config
stratml config:set <key> <value>
stratml config:get <key>
stratml config:reset
```

# StratML CLI Commands

---

## init

Create a default `config.yaml` in the current directory.

```bash
stratml init
```

Edit `dataset.path` and `dataset.target_column` before running anything else.

---

## validate-config

Check that a config file is valid before running.

```bash
stratml validate-config <config>
```

Example:

```bash
stratml validate-config config.yaml
```

---

## profile-data

Load a dataset, compute a DataProfile, and print a summary to the terminal.
Saves full JSON output to `outputs/<dataset_name>/data_profile.json`.

```bash
stratml profile-data <dataset> <target>
```

Example:

```bash
stratml profile-data data/iris.csv species
```

---

## run

Run the AutoML pipeline using a config file.

```bash
stratml run <config> [options]
```

Example:

```bash
stratml run config.yaml
```

Optional flags:

```
--path <dataset>               override dataset path from config
--mode <beginner|intermediate|expert>
--max-iter <n>                 override max iterations
--dry-run                      print resolved config without running
```

Dry run example:

```bash
stratml run config.yaml --dry-run
```

---

## doctor

Check that all required packages are installed.

```bash
stratml doctor
```

---

## Argument Style

All commands use **positional arguments** — no `--input` or `--config` flags needed.

| Command | Positional args |
|---|---|
| `run` | `config` |
| `validate-config` | `config` |
| `profile-data` | `dataset`, `target` |
| `init` | none |
| `doctor` | none |

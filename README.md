# spider_cw

## Requirements

- **Python**: see `requires-python` in `pyproject.toml` for the supported version.
- **[uv](https://docs.astral.sh/uv/)** (strongly recommended): used for dependency management, virtual environments and running the project.

## Getting started

### 1. Install uv

```sh
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Other install methods (Homebrew, pipx, etc.) are listed in the [uv docs](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Clone and set up the environment

```sh
git clone <repo-url>
cd spider_cw
uv sync
```

`uv sync` creates a virtual environment in `.venv/` (if one doesn't already exist), installs the exact dependency versions pinned in `uv.lock`, and installs the project itself in editable mode.

### 3. Run the project

```sh
uv run spider-cw
```

`uv run` always runs inside the project's virtual environment, so you don't need to activate it manually.

## Virtual environments

**Always work inside the project's virtual environment.** Do not install packages into your global Python.

- With uv, this is handled for you: `uv sync` and `uv run` both use `.venv/` automatically.
- If you'd rather activate it manually (e.g. for your editor or a plain `python` command):

    ```sh
    # macOS / Linux
    source .venv/bin/activate

    # Windows (PowerShell)
    .venv\Scripts\Activate.ps1
    ```

- Point your editor/IDE at `.venv/` as the Python interpreter.
- Do **not** commit `.venv/`. It should be listed in `.gitignore`.

### Not using uv?

uv is recommended, but if you can't use it, plain `venv` + `pip` works:

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e .
spider-cw
```

Note that this ignores `uv.lock`, so you may end up with slightly different dependency versions to everyone else. If something works for you but not for others (or vice versa), switch to uv first when debugging.

## Managing dependencies

Use uv commands rather than editing `pyproject.toml` by hand:

```sh
uv add <package>            # add a runtime dependency
uv add --dev <package>      # add a dev-only dependency (e.g. pytest)
uv remove <package>         # remove a dependency
uv sync                     # install/update the environment to match uv.lock
```

Commit both `pyproject.toml` and `uv.lock` whenever dependencies change. After pulling changes from others, run `uv sync` to stay up to date.

## Project structure

This project uses the [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/):

```
spider_cw
├── pyproject.toml          # project metadata, dependencies, entry points
├── README.md
├── uv.lock                 # pinned dependency versions (commit this)
├── src
│   └── spider_cw           # the actual package
│       ├── __init__.py
│       └── main.py         # program entry point
└── tests                   # tests live outside src/
```

- All source code goes in `src/spider_cw/`.
- Import between modules using the package name, e.g. `from spider_cw.parser import parse_page`.
- Don't run files inside `src/` directly (`python src/spider_cw/main.py`); relative imports will break. Use `uv run spider-cw` instead.

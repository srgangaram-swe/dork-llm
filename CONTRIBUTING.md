# Contributing to AxiomStack

Thanks for your interest. AxiomStack is a portfolio/research project, but it is
built to a professional bar and contributions are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"     # or a subset: .[train] / .[rag] / .[eval] / .[serve]
pre-commit install
```

## Before you push

The CI gate covers Ruff, Black, mypy, pytest on Python 3.11–3.13, frontend unit
tests, and Playwright browser integration. Run the local gates with:

```bash
make check        # lint + black --check + typecheck + tests
make smoke        # end-to-end platform self-test (no GPU, ~1 min)
```

Formatting and linting are automated:

```bash
make format       # black + ruff --fix
```

## Conventions

- **Style:** black (line length 100) + ruff; full type hints; Google-style docstrings.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`). Keep them small and logical.
- **Branches:** `feat/<slug>`, `fix/<slug>`, `docs/<slug>`; target `dev`.
- **Tests:** new behavior needs a test. Offline by default; gate heavy paths with
  `pytest.importorskip(...)` and the `@pytest.mark.torch` / `slow` markers.
- **Local-first:** deterministic test backends must be explicit. Never silently
  substitute a mock for a requested trained model.
- **No data dumps:** only small public/synthetic data. Never commit datasets,
  checkpoints, vector stores, or reports (see `.gitignore`).

## Project layout

See [docs/architecture.md](docs/architecture.md). Core code lives in `dork/`,
runnable entry points in `scripts/`, apps in `apps/`, docs in `docs/`.

## Reporting bugs / requesting features

Open a GitHub issue using the templates in
[.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE). Include the command you ran,
what you expected, and what happened (with logs where possible).

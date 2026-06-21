# Contributing to OpenDataSci

Thanks for your interest in contributing. OpenDataSci is an autonomous AI agent purpose-built for data science and machine learning. This document covers how to report bugs, propose features, and get a working dev environment so your PR can land cleanly.

---

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Reporting bugs](#reporting-bugs)
- [Proposing features](#proposing-features)
- [Development setup](#development-setup)
- [Code quality](#code-quality)
- [Running tests](#running-tests)
- [Submitting a pull request](#submitting-a-pull-request)
- [Commit style](#commit-style)
- [License](#license)

---

## Ways to contribute

- **Bug reports** — open an issue with a minimal reproduction.
- **Feature requests** — open an issue describing the use case before building anything.
- **Pull requests** — bug fixes, documentation improvements, and small well-scoped features. For larger changes, open an issue first so we can align on approach.
- **Documentation** — typo fixes, clearer examples, missing API docs.

---

## Reporting bugs

Open a GitHub issue and include:

1. What you did (ideally a minimal script or TUI invocation).
2. What you expected to happen.
3. What actually happened (error message, traceback, unexpected output).
4. Your environment: OS, Python version, `open-data-sci` version (`pip show open-data-sci`), provider.

---

## Proposing features

Open an issue before writing code. Describe the use case and why the current behaviour falls short. This lets us discuss design trade-offs before you invest time in an implementation.

---

## Development setup

**Requirements:**
- Python 3.12 (exact — `requires-python = ">=3.12"`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- System dependencies for the sandbox runtime (see below)

```bash
# 1. Clone the repo
git clone https://github.com/f4roukb/open-data-sci.git
cd open-data-sci/libs/open-data-sci

# 2. Install system-level sandbox dependencies
#    (ripgrep on macOS; bubblewrap + socat + ripgrep on Linux)
make install-system-dependencies

# 3. Install Python dependencies (includes dev extras)
make install-dev

# 4. Set up your API keys
cp .env.example .env
# Edit .env and fill in at least ANTHROPIC_API_KEY
```

That's it. `uv` creates and manages a virtual environment automatically — no `python -m venv` needed.

---

## Code quality

All checks must pass before a PR can merge. Run them locally before pushing:

```bash
make static-checks   # format check + lint + type check (read-only, mirrors CI)
make all             # format + lint-fix + type check (writes changes)
```

The three tools in play:

| Tool | What it checks |
|------|---------------|
| `ruff format` | Code formatting (line length 100) |
| `ruff check` | Linting (E, F, I, N, W rule sets) |
| `mypy` | Static typing (strict mode) |

New code must be fully typed. `mypy` runs in strict mode — `# type: ignore` comments need a comment explaining why.

---

## Running tests

```bash
make test-unit   # fast, no API keys required
make test-component    # component tests — no API keys required
make test        # both
```

Unit tests live in `tests/unit/` and must not call any external API or network service. Component tests live in `tests/component/` and wire real internal components together with external boundaries (LLM, sandbox) replaced by stubs — no API keys required.

When adding a feature, add unit tests for the logic and, where appropriate, a component test that exercises the full service path.

---

## Submitting a pull request

1. **Fork** the repo and create a branch from `main`.
2. Make your change. Keep the scope focused — one logical change per PR.
3. Run `make all` and `make test` locally.
4. Open a PR against `main`. Fill in what changed and why.
5. A maintainer will review and may request changes. Address feedback with new commits (don't force-push during review).

PRs that fail CI will not be merged. If CI is red for a reason unrelated to your change, note that in the PR description.

---

## Commit style

Use the conventional commits format:

```
<type>: <short summary>

<optional body>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `breaking`.

Examples:
```
feat: add vLLM provider support
fix: handle empty DataFrame in dataset profiling
perf: reduce memory usage in concurrent worker aggregation
docs: add component test setup instructions
```

Keep the summary line under 72 characters. The body is optional but useful for non-obvious changes.

---

## Releasing

_For maintainers._ The project uses two long-lived branches: `main` (active
development) and `release` (what has been shipped). Releasing is deliberately a
manual decision with the mechanical parts automated.

1. **Bump the version** in `libs/open-data-sci/pyproject.toml` (`project.version`)
   on `main`, following [Semantic Versioning](https://semver.org/). Update
   `CHANGELOG.md` in the same change.
2. **Open a PR from `main` into `release`** and merge it once CI is green. On
   merge, the [Tag Release](.github/workflows/tag.yaml) workflow creates the
   `v{version}` tag automatically. If the version was not bumped, the tag already
   exists and the workflow skips quietly.
3. **Publish the tag.** From the Actions tab, run the
   [Publish](.github/workflows/publish.yaml) workflow and pass the tag (e.g.
   `v0.2.0`). It verifies the tag matches `pyproject.toml`, builds the package,
   publishes to PyPI, and creates the GitHub Release. Publishing is decoupled
   from tagging, so a tag can exist before it is on PyPI, and a failed publish
   can be re-run on the same tag.

---

## License

By contributing, you agree that your contributions will be licensed under the same [Apache License 2.0](LICENSE) that covers this project.

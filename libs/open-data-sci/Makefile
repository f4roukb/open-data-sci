.PHONY: help install install-dev install-system-dependencies format format-check lint lint-fix type-check deadcode static-checks all clean test test-unit test-component docs docs-serve

PACKAGE = opendatasci

help:
	@echo "Available targets:"
	@echo "  install        Install package in production mode (EXTRAS=aws,gemini,...)"
	@echo "  install-dev    Install package with dev dependencies (EXTRAS=aws,gemini,...)"
	@echo "  install-system-dependencies Install system-level dependencies required by sandbox-runtime (SRT)"
	@echo "  format         Run ruff formatter (writes)"
	@echo "  format-check   Run ruff formatter in check mode (read-only)"
	@echo "  lint           Run ruff linter"
	@echo "  lint-fix       Run ruff linter with auto-fix"
	@echo "  type-check     Run mypy type checker"
	@echo "  deadcode       Find dead code with vulture (60%% confidence)"
	@echo "  static-checks  Run format-check, lint, and type-check"
	@echo "  all            Format, lint-fix, and type-check (writes)"
	@echo "  test           Run all tests (unit + component)"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-component       Run component tests only"
	@echo "  docs           Build documentation to site/"
	@echo "  docs-serve     Serve documentation locally with live-reload"
	@echo "  clean          Remove build artifacts and caches"

EXTRAS ?=
_extra_flags = $(foreach e,$(EXTRAS),--extra $(e))

install:
	uv sync $(_extra_flags)

install-dev:
	uv sync --extra dev $(_extra_flags)

# sandbox-runtime (SRT) relies on native OS sandboxing binaries.
# macOS needs ripgrep; Linux also needs bubblewrap and socat.
install-system-dependencies:
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Installing macOS system deps for SRT (ripgrep)..."; \
		brew install ripgrep; \
	elif [ "$$(uname)" = "Linux" ]; then \
		echo "Installing Linux system deps for SRT (bubblewrap socat ripgrep)..."; \
		if command -v apt-get >/dev/null 2>&1; then \
			sudo apt-get install -y bubblewrap socat ripgrep; \
		elif command -v dnf >/dev/null 2>&1; then \
			sudo dnf install -y bubblewrap socat ripgrep; \
		elif command -v pacman >/dev/null 2>&1; then \
			sudo pacman -S --noconfirm bubblewrap socat ripgrep; \
		else \
			echo "Unsupported package manager. Install bubblewrap, socat, and ripgrep manually."; \
			exit 1; \
		fi; \
	else \
		echo "Unsupported platform: $$(uname)"; \
		exit 1; \
	fi

format:
	uv run ruff format $(PACKAGE)

format-check:
	uv run ruff format --check $(PACKAGE)

lint:
	uv run ruff check $(PACKAGE)

lint-fix:
	uv run ruff check --fix $(PACKAGE)

type-check:
	uv run mypy $(PACKAGE)

deadcode:
	uv run vulture $(PACKAGE) --min-confidence 60

static-checks: format-check lint type-check

all: format lint-fix type-check

test: test-unit test-component

test-unit:
	uv run pytest tests/unit -n auto --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html:coverage/unit --cov-report=xml:coverage/unit.xml --junitxml=coverage/unit-junit.xml -o junit_family=legacy

test-component:
	uv run pytest tests/component -n auto --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html:coverage/component --cov-report=xml:coverage/component.xml --junitxml=coverage/component-junit.xml -o junit_family=legacy

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf site/
	rm -rf .uv-cache/
	rm -rf .benchmarks/
	rm -rf coverage/
	rm -f .coverage .coverage.*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docs:
	uv run mkdocs build --strict -f mkdocs.yaml

docs-serve:
	uv run mkdocs serve -f mkdocs.yaml

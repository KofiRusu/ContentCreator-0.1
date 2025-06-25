# ContentCreator Makefile
# Automated development and git pipeline commands

.PHONY: help setup install test lint format security git-setup git-commit git-push git-auto clean build docs

# Default target
help:
	@echo "ContentCreator Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Setup development environment"
	@echo "  install        - Install dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  security       - Run security checks"
	@echo "  clean          - Clean build artifacts"
	@echo "  build          - Build package"
	@echo "  docs           - Generate documentation"
	@echo ""
	@echo "Git Automation Commands:"
	@echo "  git-setup      - Setup git repository and remote"
	@echo "  git-commit     - Add and commit changes"
	@echo "  git-push       - Push changes to remote"
	@echo "  git-auto       - Complete automated git pipeline"
	@echo "  git-force      - Force push changes"
	@echo "  git-test       - Run tests before git operations"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make git-auto MESSAGE='Add new feature'"
	@echo "  make git-test"
	@echo "  make git-force"

# Setup development environment
setup:
	@echo "Setting up development environment..."
	@chmod +x scripts/git_pipeline.sh
	@chmod +x scripts/git_auto.py
	@python -m pip install --upgrade pip
	@if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	@pip install pytest pytest-cov black isort flake8 bandit safety
	@echo "Development environment setup complete!"

# Install dependencies
install:
	@echo "Installing dependencies..."
	@if [ -f pyproject.toml ] && command -v poetry >/dev/null; then \
		echo "Using Poetry..."; \
		poetry install; \
	elif [ -f requirements.txt ]; then \
		echo "Using pip..."; \
		pip install -r requirements.txt; \
	else \
		echo "No dependency files found!"; \
	fi

# Run tests
test:
	@echo "Running tests..."
	@if [ -f pyproject.toml ] && command -v poetry >/dev/null; then \
		poetry run pytest src/tests/ -v; \
	else \
		python -m pytest src/tests/ -v; \
	fi

# Run linting
lint:
	@echo "Running linting..."
	@flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code
format:
	@echo "Formatting code..."
	@black src/
	@isort src/
	@echo "Code formatting complete!"

# Run security checks
security:
	@echo "Running security checks..."
	@bandit -r src/ -f txt || true
	@safety check || true

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

# Build package
build:
	@echo "Building package..."
	@python -m build
	@echo "Build complete!"

# Generate documentation
docs:
	@echo "Generating documentation..."
	@if command -v sphinx-build >/dev/null; then \
		sphinx-build -b html docs/ docs/_build/html; \
	else \
		echo "Sphinx not installed. Install with: pip install sphinx"; \
	fi

# Git setup
git-setup:
	@echo "Setting up git repository..."
	@if [ -z "$(USERNAME)" ]; then \
		echo "Error: USERNAME not provided. Use: make git-setup USERNAME=yourusername"; \
		exit 1; \
	fi
	@python scripts/git_auto.py --setup --username $(USERNAME)

# Git commit
git-commit:
	@echo "Committing changes..."
	@if [ -n "$(MESSAGE)" ]; then \
		python scripts/git_auto.py --message "$(MESSAGE)"; \
	else \
		python scripts/git_auto.py; \
	fi

# Git push
git-push:
	@echo "Pushing changes..."
	@git push origin main

# Complete automated git pipeline
git-auto:
	@echo "Running automated git pipeline..."
	@if [ -z "$(USERNAME)" ]; then \
		echo "Error: USERNAME not provided. Use: make git-auto USERNAME=yourusername"; \
		exit 1; \
	fi
	@if [ -n "$(MESSAGE)" ]; then \
		python scripts/git_auto.py --username $(USERNAME) --message "$(MESSAGE)"; \
	else \
		python scripts/git_auto.py --username $(USERNAME); \
	fi

# Force push
git-force:
	@echo "Force pushing changes..."
	@if [ -z "$(USERNAME)" ]; then \
		echo "Error: USERNAME not provided. Use: make git-force USERNAME=yourusername"; \
		exit 1; \
	fi
	@python scripts/git_auto.py --username $(USERNAME) --force

# Run tests before git operations
git-test:
	@echo "Running tests before git operations..."
	@if [ -z "$(USERNAME)" ]; then \
		echo "Error: USERNAME not provided. Use: make git-test USERNAME=yourusername"; \
		exit 1; \
	fi
	@python scripts/git_auto.py --username $(USERNAME) --test

# Quick git operations (commonly used combinations)
quick-commit:
	@$(MAKE) format
	@$(MAKE) git-commit MESSAGE="$(MESSAGE)"

quick-push:
	@$(MAKE) format
	@$(MAKE) test
	@$(MAKE) git-auto USERNAME=$(USERNAME) MESSAGE="$(MESSAGE)"

# Development workflow
dev-check:
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) test
	@$(MAKE) security

# Full pipeline
full-pipeline:
	@$(MAKE) dev-check
	@$(MAKE) git-auto USERNAME=$(USERNAME) MESSAGE="$(MESSAGE)" 
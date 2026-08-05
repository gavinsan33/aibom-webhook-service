# Bootstrap only. Everything else (build/test/deploy) lives in the justfile —
# run `just` once `just` itself is installed.
.DEFAULT_GOAL := help

.PHONY: help install-just

help:
	@echo "New here? Run 'make install-just', then 'just' to see available recipes."

install-just:
	@if command -v just >/dev/null 2>&1; then \
		echo "just is already installed ($$(just --version))"; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install just; \
	elif command -v cargo >/dev/null 2>&1; then \
		cargo install just; \
	else \
		curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin; \
		echo "Installed just to ~/.local/bin — add it to your PATH if it isn't already."; \
	fi
	@echo "Run 'just' to see available recipes."

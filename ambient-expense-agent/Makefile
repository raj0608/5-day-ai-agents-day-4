# Makefile for Ambient Expense Approval Agent

.PHONY: run test install lint generate-traces grade eval help

help:
	@echo "Ambient Expense Approval Agent Makefile"
	@echo "---------------------------------------"
	@echo "make run             - Start ambient web service locally on port 8080"
	@echo "make install         - Install dependencies via uv"
	@echo "make generate-traces - Generate trace artifacts from dataset scenarios"
	@echo "make grade           - Run LLM-as-judge evaluation grading"
	@echo "make eval            - Run generate-traces and grade targets"
	@echo "make test            - Run pytest unit and integration tests"
	@echo "make lint            - Run code quality and lint checks"

run:
	uv run python -m uvicorn app.fast_api_app:app --host 0.0.0.0 --port 8080 --reload

install:
	uv sync

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	uv run python tests/eval/run_grading.py

eval: generate-traces grade

test:
	uv run pytest

lint:
	uv run ruff check .

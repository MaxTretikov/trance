# Development

Use `xonsh` and `uv` for project commands. This package is a `src`-layout Python
library. Keep provider discovery isolated behind small source adapters, keep
credential values out of logs and fixtures. The core package is provider
neutral and has no model SDK dependency. Optional adapters are imported lazily
and installed directly by the application when needed; do not add provider
SDKs as `trance` extras. Examples include `uv add pydantic-ai-slim[openai]`,
`uv add litellm`, `uv add langchain-openai`, `uv add langchain-anthropic`,
`uv add llama-index-llms-openai`, `uv add llama-index-llms-openai-like`, and
`uv add openai`.

Set up the development environment with `uv sync --extra dev`. Run the suite
with `uv run pytest` and lint with `uv run ruff check .`.

Do not commit local credentials, environment files, virtual environments, or
generated build and test artifacts.

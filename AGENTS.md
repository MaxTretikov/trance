# Development

Use `xonsh` and `uv` for project commands. This package is a `src`-layout Python
library. Keep provider discovery isolated behind small source adapters, keep
credential values out of logs and fixtures, and keep optional provider SDKs in
separate extras. The base dependency uses Pydantic AI's OpenAI adapter; install
a named provider extra only when an adapter needs its native SDK (for example,
`uv sync --extra dev --extra anthropic`).

Set up the development environment with `uv sync --extra dev`. Run the suite
with `uv run pytest` and lint with `uv run ruff check .`.

Do not commit local credentials, environment files, virtual environments, or
generated build and test artifacts.

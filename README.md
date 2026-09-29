# trance

`trance` discovers a limited set of documented credentials and signed-in
coding tools already configured on your machine, then builds model objects
using [Pydantic AI](https://ai.pydantic.dev/). It is a small Python library
for applications that want to reuse supported local provider setups.

Consumer subscriptions do not automatically grant general API access. A
credential being present does not establish that a provider permits using it
from arbitrary software. `trance` scans only documented local auth state. It
does not search private browser stores or undocumented application stores;
raw token strings are not returned or logged, although model objects may retain
auth internally for requests. Supported CLI adapters may privately stage
documented saved credentials for a request. Gemini CLI OAuth
is surfaced only with a warning because its [terms prohibit third-party
access to the service through its OAuth session and may suspend or terminate
accounts](https://github.com/google-gemini/gemini-cli/blob/main/docs/resources/tos-privacy.md).

## Coverage at a glance

| Product | Current supported path |
| --- | --- |
| OpenAI Codex | Subscription authentication delegated to the documented Codex CLI/Pydantic AI integration |
| Claude Code | Subscription authentication delegated to the documented Claude CLI, including a saved CLI login or `CLAUDE_CODE_OAUTH_TOKEN` |
| Grok Build consumer | Optional integration requiring the installed official `grok` CLI and an authenticated saved session; the CLI adapter validates the session lazily when a request runs |
| Grok API (xAI) | `XAI_API_KEY` through the regular API-key registry; this is a separate API credential path |
| Gemini CLI consumer | Optional saved-login integration using the installed `gemini` CLI and `~/.gemini/oauth_creds.json`; discovery emits the documented terms warning and does not provide an opt-in switch |
| Gemini API | `GOOGLE_API_KEY` or `GEMINI_API_KEY` through the regular API-key registry; this is a separate API credential path |
| Sourcegraph Cody | Optional saved account login delegated to the installed `cody` CLI |
| OpenCode | Optional provider-keyed `auth.json` entries for known defaults and model overrides; account/API credentials are delegated to an isolated, no-tools `opencode` CLI |
| AWS Bedrock | Optional local environment, profile, or SSO credential chain; uses the `bedrock` extra |
| Google Vertex AI | Optional local Application Default Credentials (ADC); uses the `google` extra |

Grok Build and the xAI API are separate credential paths. A consumer account
or subscription does not by itself provide an xAI API credential. Gemini CLI
OAuth is a separate, warned integration with Google's terms risk; Gemini API
keys remain a separate credential path.

## Install

For a checkout, install the package and development dependencies with:

```sh
uv sync --extra dev
```

The core dependency includes Pydantic AI's OpenAI support. Optional extras for
provider integrations supported by `trance` are `anthropic`, `bedrock`,
`cerebras`, `cohere`, `crusoe`, `google`, `groq`, `huggingface`, `mistral`,
`openrouter`, and `xai`. The fixed compatible adapters, including Crusoe, use
the core OpenAI-compatible model path, so the `crusoe` extra is not required
for them.

## Quick start

Sign in with a supported local CLI or configure a documented provider key,
then scan:

```python
from trance.discovery import scan
from pydantic_ai import Agent

found = scan()
for item in found:
    print(item.provider, item.auth_kind, item.source, item.model_name)

# Use a discovered Pydantic AI Model with Agent. Calling run() sends a request.
agents = [Agent(model=item.model) for item in found]
```

`scan()` returns `list[FoundModel]`. Each record identifies the provider,
authentication kind, discovery source, and model name alongside a Pydantic AI
`Model`. It constructs objects without sending a request. It does not check
whether a credential is currently valid, permitted for a particular use, or
within quota. A `FoundModel` represents a successfully constructed model;
source adapters that cannot be safely connected to a supported model are
skipped in the default non-strict mode. Set `strict=True` to raise on a source
or model construction failure.

You can restrict discovery by provider ID, or pass an environment mapping and
home directory for controlled runs:

```python
from pathlib import Path
from trance.discovery import scan

found = scan(
    environ={"OPENAI_API_KEY": "sk-example-not-a-real-key"},
    home=Path("/home/me"),
    providers={"openai"},
)
```

Avoid printing or logging the model object: it may retain provider
authentication material for requests. `FoundModel` and candidate
representations redact their model and secret fields. The `source` field
describes where a credential was found and is safe to display.

## Discovery sources

`scan()` currently registers these adapters. Subscription and account
integrations are scanned first; ordinary API credentials are scanned after
them. A subscription login is an integration with the vendor's documented
account or CLI flow. It is not a general-purpose API key.

| Provider or source | What is read | Result and model factory |
| --- | --- | --- |
| OpenAI Codex | Bounded metadata from `~/.codex/auth.json` or the file under `CODEX_HOME` (`auth_mode` and the `tokens` shape); token values are never exposed | Subscription candidate; `OpenAICodexModel` resolves the credential |
| GitHub Copilot | `GITHUB_COPILOT_API_KEY`, `GITHUB_COPILOT_API_TOKEN`, `COPILOT_GITHUB_TOKEN`, or a saved `gh` login | Subscription candidate; `GitHubCopilotModel` |
| Claude Code | `CLAUDE_CODE_OAUTH_TOKEN`, or a saved first-party login confirmed by `claude auth status` plus `~/.claude/.credentials.json` (or `CLAUDE_CONFIG_DIR/.credentials.json`) | Subscription candidate; text-only Claude CLI model, defaulting to `sonnet` |
| Poe | `POE_API_KEY` | Subscription candidate; OpenAI-compatible model at Poe's endpoint |
| MiniMax Coding Plan | `MINIMAX_API_KEY` with the `sk-cp-` prefix and an allowed `MINIMAX_API_HOST` | Subscription candidate; OpenAI-compatible model |
| Mistral Vibe | `MISTRAL_API_KEY` or `~/.vibe/.env:MISTRAL_API_KEY` | API-key candidate; Mistral model |
| Qwen Code Coding Plan | `BAILIAN_CODING_PLAN_API_KEY` or supported `~/.qwen/settings.json`, with an `sk-sp-` key | Subscription candidate; OpenAI-compatible model at an allowed Coding Plan endpoint |
| Hugging Face Hub | Saved login resolved by `huggingface_hub.get_token()` in the real process context | Account candidate; Hugging Face model |
| Grok Build consumer | Installed `grok` executable and bounded metadata from a non-symlink `~/.grok/auth.json` (or `GROK_HOME/auth.json`) confirming an OIDC record; token values are not retained | Optional account candidate; isolated, text-only Grok CLI model; session validity is checked lazily by the CLI when a request runs, and refreshed auth is promoted back safely |
| Gemini CLI consumer | Installed `gemini` executable and bounded metadata from `~/.gemini/oauth_creds.json` (or `GEMINI_CLI_HOME/.gemini/oauth_creds.json`) confirming a refresh token; emits a terms warning | Optional account candidate; isolated, text-only Gemini CLI model; no opt-in bypass for the warning |
| Sourcegraph Cody | Installed `cody` executable and a successful local `cody auth whoami` check with dedicated PAT variables removed | Account candidate; text-only Cody CLI model |
| OpenCode | Installed `opencode` executable and bounded, provider-keyed `~/.local/share/opencode/auth.json` metadata; recognized OAuth, API-key, and well-known entries only | Account or API-key candidate; isolated `opencode --pure` text-only model with tools disabled; OpenCode manages its own provider-specific OAuth flow |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, or bounded AWS profile credentials/SSO configuration plus a region | Account candidate; Bedrock model through the optional `bedrock` integration |
| Google Vertex AI | Bounded local ADC metadata from `GOOGLE_APPLICATION_CREDENTIALS` or gcloud ADC, plus a project and optional location | Account candidate; Vertex model through the optional `google` integration |
| Z.AI | `ZAI_API_KEY` for the general API and `ZAI_CODING_PLAN_API_KEY` for the Coding Plan endpoint | General API-key candidate or Coding Plan subscription candidate; OpenAI-compatible models at separate official endpoints |
| Fixed compatible API registry | `MOONSHOT_API_KEY`, `NEBIUS_API_KEY`, `DEEPINFRA_TOKEN`/`DEEPINFRA_API_KEY`, `NVIDIA_API_KEY`, `NOVITA_API_KEY`, `AIML_API_KEY`, `OVH_AI_ENDPOINTS_ACCESS_TOKEN`, `HELICONE_API_KEY`, `REQUESTY_API_KEY`, `FEATHERLESS_API_KEY`, `HYPERBOLIC_API_KEY`, `CRUSOE_API_KEY`, `SILICONFLOW_API_KEY`, `VENICE_API_KEY`, `CHUTES_API_KEY`, `AKASH_API_KEY`, `SCW_SECRET_KEY`, `FRIENDLI_API_KEY`, `CLARIFAI_PAT`, `MODEL_API_KEY`, `PARASAIL_API_KEY`, and `NSCALE_API_KEY` | API-key candidates; OpenAI-compatible models at fixed provider endpoints |
| Validated cloud API registry | Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`), Cloudflare Workers AI (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_KEY`), Databricks (`DATABRICKS_TOKEN`, `DATABRICKS_HOST`, `DATABRICKS_MODEL`), and DashScope (`DASHSCOPE_API_KEY`, optional `DASHSCOPE_BASE_URL`) | API-key candidates; OpenAI-compatible models with provider-owned HTTPS endpoints |
| Generic API-key registry | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `CEREBRAS_API_KEY`, `HF_TOKEN`/`HUGGINGFACE_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `FIREWORKS_API_KEY`, `TOGETHER_API_KEY`, `PERPLEXITY_API_KEY`, `SAMBANOVA_API_KEY` | API-key candidates; the Pydantic AI provider factory, with documented OpenAI-compatible fallbacks where available |

The fixed compatible registry currently covers 22 API-key providers, including
AkashML, Scaleway, Friendli, Clarifai, Meta Model API, Parasail, and Nscale.
These credentials are ordinary API credentials, not subscription or account-
login integrations.

The model construction path supports the registered provider IDs: API and
account integrations use `trance.models.build_model()`, while Claude Code,
Grok Build, Gemini CLI, Cody, and OpenCode use text-only CLI adapters. It creates Pydantic AI model objects
without sending a request and loads optional provider dependencies lazily. Use
`TRANCE_MODEL_<PROVIDER>` variables to override defaults where the source
supports them. To use the optional Grok Build integration, install the
official `grok` CLI and authenticate it with `grok login` (or
`grok login --device-auth` on a headless machine). Discovery only checks the
CLI and bounded OIDC metadata in the saved `auth.json`; it does not claim that
the session is valid until the CLI handles a model request. The optional Gemini
CLI integration similarly requires the official `gemini` CLI and a saved
`oauth_creds.json`; discovery emits its terms warning whenever it finds one.
The optional Cody and OpenCode integrations likewise require their respective
CLIs to be installed; discovery performs only their documented local login or
metadata checks and does not make a model request.

Discovery performs no live model API validation and does not establish that a
credential is valid, permitted for a particular use, or within quota. It may
inspect explicitly supported local files, including bounded Codex auth metadata,
Grok OIDC metadata, Gemini OAuth metadata, OpenCode auth metadata, bounded AWS
profiles, and local Vertex ADC metadata, run `cody auth whoami` and `gh auth
status` and (when materializing a Copilot candidate) `gh auth token`, resolve a
saved Hugging Face token, and run `claude auth status` plus a credentials-file
existence check to detect a saved Claude subscription login. The Claude CLI is not used for a model request until a
Pydantic AI request is made. Its adapter supports text requests only; it does
not support Pydantic AI tools or structured output.

Cursor, Kiro, Tabnine, and Kimi Code adapters remain source-only and are
intentionally excluded from `scan()` results because they have no safe,
supported model bridge. In particular, their CLIs can expose hooks,
integrations, or credential flows that cannot be constrained to the supported
text-only request path.

Read the vendors' documentation for their authentication and use conditions:
[Codex CLI sign-in](https://help.openai.com/en/articles/11381614-api-codex-cli-and-sign-in-with-chatgpt),
[Copilot CLI authentication](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli),
[Claude Code authentication](https://code.claude.com/docs/en/iam),
[Grok Build overview](https://docs.x.ai/build/overview),
[Grok Build CLI reference](https://docs.x.ai/build/cli/reference),
[Sourcegraph Cody CLI](https://sourcegraph.com/docs/cody/overview),
[OpenCode documentation](https://opencode.ai/docs/),
[Amazon Bedrock authentication](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html),
[Google Vertex AI authentication](https://cloud.google.com/docs/authentication/application-default-credentials),
[Z.AI documentation](https://docs.z.ai/),
[Qwen Code Coding Plan](https://github.com/QwenLM/qwen-code/blob/main/docs/users/configuration/model-providers.md),
[Hugging Face Hub login](https://huggingface.co/docs/huggingface_hub/quick-start#login),
[Poe API keys](https://creator.poe.com/docs/external-applications),
[MiniMax API](https://platform.minimax.io/docs/api-reference),
[Mistral Vibe](https://docs.mistral.ai/capabilities/vibe/).

## Development

This repository uses `uv` and `xonsh`:

```sh
uv sync --extra dev
uv run pytest
uv run ruff check .
```

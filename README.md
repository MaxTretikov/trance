# trance

[![Tests](https://github.com/MaxTretikov/trance/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/MaxTretikov/trance/actions/workflows/tests.yml)

`trance` discovers a limited set of documented credentials and signed-in
coding tools already configured on your machine. It returns provider-neutral
candidate records and lets an application choose an adapter such as
[Pydantic AI](https://ai.pydantic.dev/) or [LiteLLM](https://docs.litellm.ai/)
only when that integration is needed. The optional adapters cover five widely
used Python LLM libraries: Pydantic AI, LiteLLM, LangChain, LlamaIndex, and the
official OpenAI Python SDK. This is a broad ecosystem choice, not a verified
ranking of the exact five most popular libraries.

Consumer subscriptions do not automatically grant general API access. A
credential being present does not establish that a provider permits using it
from arbitrary software. `trance` scans only documented local auth state. It
does not search private browser stores or undocumented application stores.
API-key candidates also retain their direct `secret` field for adapter
compatibility; it is redacted from normal representations. The explicit,
lazy `credentials` accessor provides the normalized view when credential
material is available. Values are never included in normal representations or
logs, although adapted model objects may retain auth internally for requests.
Supported CLI adapters may privately stage documented saved credentials for a
request. Gemini CLI OAuth
is surfaced only with a warning because its [terms prohibit third-party
access to the service through its OAuth session and may suspend or terminate
accounts](https://github.com/google-gemini/gemini-cli/blob/main/docs/resources/tos-privacy.md).

## Coverage at a glance

![Subscription authentication delegated to the documented Codex CLI integration](https://img.shields.io/badge/OpenAI_Codex-supported-brightgreen)<br>
![Subscription authentication delegated to the documented Claude CLI, including a saved CLI login or `CLAUDE_CODE_OAUTH_TOKEN`](https://img.shields.io/badge/Claude_Code-supported-brightgreen)<br>
![Optional integration requiring the installed official `grok` CLI and an authenticated saved session; the CLI adapter validates the session lazily when a request runs](https://img.shields.io/badge/Grok_Build_consumer-supported-brightgreen) ![`XAI_API_KEY` through the regular API-key registry; this is a separate API credential path](https://img.shields.io/badge/Grok_API_(xAI)-supported-brightgreen)<br>
![Optional saved-login integration using the installed `gemini` CLI; discovery emits the documented terms warning and does not provide an opt-in switch](https://img.shields.io/badge/Gemini_CLI_consumer-supported-brightgreen) ![`GOOGLE_API_KEY` or `GEMINI_API_KEY` through the regular API-key registry; this is a separate API credential path](https://img.shields.io/badge/Gemini_API-supported-brightgreen) ![Optional local Application Default Credentials (ADC)](https://img.shields.io/badge/Google_Vertex_AI-supported-brightgreen)<br>
![Optional saved account login delegated to the installed `cody` CLI](https://img.shields.io/badge/Sourcegraph_Cody-supported-brightgreen)<br>
![Optional provider-keyed local auth metadata for known defaults and model overrides; account/API credentials are delegated to an isolated, no-tools `opencode` CLI](https://img.shields.io/badge/OpenCode-supported-brightgreen)<br>
![Optional local environment, profile, or SSO credential chain](https://img.shields.io/badge/AWS_Bedrock-supported-brightgreen)

Grok Build and the xAI API are separate credential paths. A consumer account
or subscription does not by itself provide an xAI API credential. Gemini CLI
OAuth is a separate, warned integration with Google's terms risk; Gemini API
keys remain a separate credential path.

## Install

For an installed release, add the provider-neutral core with:

```sh
uv add trance
```

For a local checkout, use `uv sync` as shown below or add it with
`uv add --editable .`.

The core package has no model SDK dependency. Install an adapter directly when
you use it:

```sh
uv add pydantic-ai-slim[openai]  # for Candidate.to_pydantic_ai()
uv add litellm                   # for Candidate.to_litellm()
uv add langchain-openai          # for Candidate.to_langchain() with OpenAI
uv add langchain-anthropic       # for Candidate.to_langchain() with Anthropic
uv add llama-index-llms-openai   # for Candidate.to_llama_index() with OpenAI
uv add llama-index-llms-openai-like  # for approved compatible endpoints
uv add openai                    # for Candidate.to_openai()
```

These are direct dependencies of the application, not `trance` extras. Install
only the integration you use. Their official documentation is available for
[Pydantic AI](https://ai.pydantic.dev/),
[LiteLLM](https://docs.litellm.ai/),
[LangChain](https://python.langchain.com/docs/integrations/chat/openai/),
[LlamaIndex OpenAI](https://docs.llamaindex.ai/en/stable/examples/llm/openai/),
[LlamaIndex OpenAI compatible](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/), and the
[OpenAI Python SDK](https://github.com/openai/openai-python).

For a checkout, install the package and development dependencies with:

```sh
uv sync --extra dev
```

The development extra includes the Pydantic AI adapter's test dependency. It
does not make Pydantic AI a runtime dependency of installed `trance` packages.

## Quick start

Sign in with a supported local CLI or configure a documented provider key,
then scan:

```python
from trance import scan

found = scan()
for item in found:
    print(item.provider, item.auth_kind, item.source, item.model_name)

# Choose an adapter explicitly. This imports Pydantic AI only at this call.
model = found[0].to_pydantic_ai()
from pydantic_ai import Agent
result = Agent(model=model).run_sync("Say hello in one sentence.")
print(result.output)
```

`scan()` returns `list[Candidate]`. Each record identifies the provider,
authentication kind, discovery source, and model name. It does not import
Pydantic AI or LiteLLM, and it does not construct model clients. Call
`candidate.to_pydantic_ai()` or
`candidate.to_litellm()` to opt into an adapter. Those methods dynamically
import their optional integration and raise a clear missing dependency error
if it is not installed. They construct objects without sending a request. They do not check
whether a credential is currently valid, permitted for a particular use, or
within quota. Source adapters that cannot be safely connected to a supported
candidate are skipped in the default non-strict mode. Set `strict=True` to
raise on a source failure.

For code that still expects Pydantic AI models, `scan_models()` and
`clients()` are compatibility helpers. They explicitly call the Pydantic AI
adapter and therefore require its direct installation.

LiteLLM is another explicit choice:

```python
from trance import scan

candidate = scan(providers={"openai"})[0]
adapter = candidate.to_litellm()
response = adapter.completion(
    [{"role": "user", "content": "Say hello in one sentence."}]
)
print(response.choices[0].message.content)
```

The other optional integrations use the same provider-neutral candidate:

```python
candidate = scan(providers={"openai"})[0]

pydantic_model = candidate.to_pydantic_ai()
litellm_client = candidate.to_litellm()
langchain_model = candidate.to_langchain()
llama_index_model = candidate.to_llama_index()
openai_client = candidate.to_openai()
async_openai_client = candidate.to_openai(asynchronous=True)
```

Each method imports its library only when called and returns that library's
native client or model object. The OpenAI adapter can return either its normal
or asynchronous client. The LiteLLM, LangChain, LlamaIndex, and OpenAI
adapters accept explicit API-key candidates for their approved provider
endpoints. They do not turn arbitrary CLI or OAuth sessions into generic
library credentials; use the documented Pydantic AI or vendor CLI integration
for those candidates. New adapters likewise require an explicit API key and a
provider endpoint that `trance` has approved.

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

Avoid printing or logging an adapted model object: it may retain provider
authentication material for requests. `FoundModel` and candidate
representations redact their model, secret, and credential fields. The
`source` field describes where a credential was found and is safe to display.

## Credential material

`Candidate.credentials` is a lazy `CredentialMaterial` view. Discovery can
therefore remain fast and provider-neutral, while an application that has a
specific reason to use credential material can request it explicitly:

```python
candidate = scan(providers={"openai"})[0]
material = candidate.credentials

# API-key candidates expose their named values through a read-only mapping.
api_key = material.values.get("api_key")

# A single-token candidate may also expose its value directly.
token = material.value
```

The available keys depend on the source. API-key candidates expose an
`api_key` value, and supported extractors can expose allowlisted values such
as Codex `access_token`, `refresh_token`, and `id_token`; Claude Code's known
OAuth access token; Gemini CLI's `access_token` and `refresh_token`; and
OpenCode's approved API or OAuth fields. Copilot's explicit token sources can
also be resolved on request.

Some sessions intentionally remain opaque. Cody credentials stay in secure
storage, and some Grok OIDC sessions remain opaque; AWS profile or SSO candidates
expose a reference rather than profile secrets, and Vertex candidates expose
an ADC reference rather than the file's credential material. Use the
corresponding adapter or vendor CLI for those candidates. The accessor does
not infer permissions from a credential, verify that a provider allows the
requested use, or imply that a subscription is equivalent to API access.

Treat values from `CredentialMaterial` as secrets: keep them in memory only as
long as needed and never print, persist, or include them in error messages.

## Discovery sources

`scan()` currently registers these source adapters. Subscription and account
integrations are scanned first; ordinary API credentials are scanned after
them. A subscription login is an integration with the vendor's documented
account or CLI flow. It is not a general-purpose API key.

| Provider or source | What is read | Result |
| --- | --- | --- |
| OpenAI Codex | Bounded local auth metadata from the configured Codex session | Subscription candidate; the lazy credentials accessor can expose allowlisted OAuth token fields |
| GitHub Copilot | `GITHUB_COPILOT_API_KEY`, `GITHUB_COPILOT_API_TOKEN`, `COPILOT_GITHUB_TOKEN`, or a saved `gh` login | Subscription candidate; supported adapters resolve the credential |
| Claude Code | `CLAUDE_CODE_OAUTH_TOKEN`, or a saved first-party login confirmed by `claude auth status` and supported local auth metadata | Subscription candidate; text-only Claude CLI model, defaulting to `sonnet`; a known OAuth access token can be extracted lazily |
| Poe | `POE_API_KEY` | Subscription candidate; OpenAI-compatible model at Poe's endpoint |
| MiniMax Coding Plan | `MINIMAX_API_KEY` with the `sk-cp-` prefix and an allowed `MINIMAX_API_HOST` | Subscription candidate; OpenAI-compatible model |
| Mistral Vibe | `MISTRAL_API_KEY` or `~/.vibe/.env:MISTRAL_API_KEY` | API-key candidate; Mistral model |
| Qwen Code Coding Plan | `BAILIAN_CODING_PLAN_API_KEY` or supported `~/.qwen/settings.json`, with an `sk-sp-` key | Subscription candidate; OpenAI-compatible model at an allowed Coding Plan endpoint |
| Hugging Face Hub | Saved login resolved by `huggingface_hub.get_token()` in the real process context | Account candidate; Hugging Face model |
| Grok Build consumer | Installed `grok` executable and bounded metadata from a non-symlink `~/.grok/auth.json` (or `GROK_HOME/auth.json`) confirming an OIDC record; token values are not retained | Optional account candidate; isolated, text-only Grok CLI model; session validity is checked lazily by the CLI when a request runs, and refreshed auth is promoted back safely |
| Gemini CLI consumer | Installed `gemini` executable and bounded local OAuth metadata confirming a refreshable session; emits a terms warning | Optional account candidate; isolated, text-only Gemini CLI model; allowlisted access and refresh tokens can be extracted lazily; no opt-in bypass for the warning |
| Sourcegraph Cody | Installed `cody` executable and a successful local `cody auth whoami` check with dedicated PAT variables removed | Account candidate; text-only Cody CLI model |
| OpenCode | Installed `opencode` executable and bounded provider-keyed local auth metadata; recognized OAuth, API-key, and well-known entries only | Account or API-key candidate; isolated `opencode --pure` text-only model with tools disabled; approved API and OAuth fields can be extracted lazily |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, or bounded AWS profile credentials/SSO configuration plus a region | Account candidate; use an explicit adapter, with profile and SSO material kept behind its reference |
| Google Vertex AI | Bounded local ADC metadata plus a project and optional location | Account candidate; use an explicit adapter, with ADC material kept behind its reference |
| Z.AI | `ZAI_API_KEY` for the general API and `ZAI_CODING_PLAN_API_KEY` for the Coding Plan endpoint | General API-key candidate or Coding Plan subscription candidate; OpenAI-compatible models at separate official endpoints |
| Fixed compatible API registry | `MOONSHOT_API_KEY`, `NEBIUS_API_KEY`, `DEEPINFRA_TOKEN`/`DEEPINFRA_API_KEY`, `NVIDIA_API_KEY`, `NOVITA_API_KEY`, `AIML_API_KEY`, `OVH_AI_ENDPOINTS_ACCESS_TOKEN`, `HELICONE_API_KEY`, `REQUESTY_API_KEY`, `FEATHERLESS_API_KEY`, `HYPERBOLIC_API_KEY`, `CRUSOE_API_KEY`, `SILICONFLOW_API_KEY`, `VENICE_API_KEY`, `CHUTES_API_KEY`, `AKASH_API_KEY`, `SCW_SECRET_KEY`, `FRIENDLI_API_KEY`, `CLARIFAI_PAT`, `MODEL_API_KEY`, `PARASAIL_API_KEY`, and `NSCALE_API_KEY` | API-key candidates; OpenAI-compatible models at fixed provider endpoints |
| Validated cloud API registry | Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`), Cloudflare Workers AI (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_KEY`), Databricks (`DATABRICKS_TOKEN`, `DATABRICKS_HOST`, `DATABRICKS_MODEL`), and DashScope (`DASHSCOPE_API_KEY`, optional `DASHSCOPE_BASE_URL`) | API-key candidates; OpenAI-compatible models with provider-owned HTTPS endpoints |
| Generic API-key registry | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `CEREBRAS_API_KEY`, `HF_TOKEN`/`HUGGINGFACE_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `FIREWORKS_API_KEY`, `TOGETHER_API_KEY`, `PERPLEXITY_API_KEY`, `SAMBANOVA_API_KEY` | API-key candidates; adapter selection and model construction happen only when requested |

The fixed compatible registry currently covers 22 API-key providers, including
AkashML, Scaleway, Friendli, Clarifai, Meta Model API, Parasail, and Nscale.
These credentials are ordinary API credentials, not subscription or account-
login integrations.

The discovery path supports the registered provider IDs and remains
provider-neutral. The optional Pydantic AI adapter uses
`trance.models.build_model()` for API and account integrations, while Claude
Code, Grok Build, Gemini CLI, Cody, and OpenCode use text-only CLI adapters.
It creates adapted model objects without sending a request and loads optional
provider dependencies lazily. Use
`TRANCE_MODEL_<PROVIDER>` variables to override defaults where the source
supports them. To use the optional Grok Build integration, install the
official `grok` CLI and authenticate it with `grok login` (or
`grok login --device-auth` on a headless machine). Discovery only checks the
CLI and bounded OIDC metadata in the saved local auth state; it does not claim
that
the session is valid until the CLI handles a model request. The optional Gemini
CLI integration similarly requires the official `gemini` CLI and a saved
local OAuth state; discovery emits its terms warning whenever it finds one.
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
 existence check to detect a saved Claude subscription login. The Claude CLI
 is not used for a model request until an adapted model is run. Its CLI adapter
 supports text requests only; it does not support Pydantic AI tools or
 structured output.

The LiteLLM adapter supports ordinary API-key candidates with approved provider
model mappings and endpoints. Delegated CLI and account candidates, including
Codex, Copilot, Claude Code, Gemini CLI, Cody, and OpenCode sessions, are not
silently converted into LiteLLM credentials. Use the Pydantic AI adapter for
the integrations it supports, or use the provider's own client when an
integration has no safe generic adapter.

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

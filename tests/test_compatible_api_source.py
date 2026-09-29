from trance.sources.compatible_api import scan


def test_scan_discovers_all_fixed_compatible_endpoints_with_model_overrides(tmp_path):
    environment = {
        "MOONSHOT_API_KEY": "moonshot-fixture",
        "NEBIUS_API_KEY": "nebius-fixture",
        "DEEPINFRA_TOKEN": "deepinfra-fixture",
        "NVIDIA_API_KEY": "nvidia-fixture",
        "NOVITA_API_KEY": "novita-fixture",
        "AIML_API_KEY": "aiml-fixture",
        "OVH_AI_ENDPOINTS_ACCESS_TOKEN": "ovh-fixture",
        "HELICONE_API_KEY": "helicone-fixture",
        "REQUESTY_API_KEY": "requesty-fixture",
        "FEATHERLESS_API_KEY": "featherless-fixture",
        "HYPERBOLIC_API_KEY": "hyperbolic-fixture",
        "CRUSOE_API_KEY": "crusoe-fixture",
        "SILICONFLOW_API_KEY": "siliconflow-fixture",
        "VENICE_API_KEY": "venice-fixture",
        "CHUTES_API_KEY": "chutes-fixture",
        "AKASH_API_KEY": "akash-fixture",
        "SCW_SECRET_KEY": "scaleway-fixture",
        "FRIENDLI_API_KEY": "friendli-fixture",
        "CLARIFAI_PAT": "clarifai-fixture",
        "MODEL_API_KEY": "meta-model-fixture",
        "PARASAIL_API_KEY": "parasail-fixture",
        "NSCALE_API_KEY": "nscale-fixture",
        "TRANCE_MODEL_NVIDIA": "custom-nvidia-model",
    }

    candidates = scan(environment, tmp_path)

    assert [item.provider for item in candidates] == [
        "moonshot",
        "nebius",
        "deepinfra",
        "nvidia",
        "novita",
        "aimlapi",
        "ovhcloud",
        "helicone",
        "requesty",
        "featherless",
        "hyperbolic",
        "crusoe",
        "siliconflow",
        "venice",
        "chutes",
        "akashml",
        "scaleway",
        "friendli",
        "clarifai",
        "meta-model-api",
        "parasail",
        "nscale",
    ]
    assert [item.secret for item in candidates] == [
        "moonshot-fixture",
        "nebius-fixture",
        "deepinfra-fixture",
        "nvidia-fixture",
        "novita-fixture",
        "aiml-fixture",
        "ovh-fixture",
        "helicone-fixture",
        "requesty-fixture",
        "featherless-fixture",
        "hyperbolic-fixture",
        "crusoe-fixture",
        "siliconflow-fixture",
        "venice-fixture",
        "chutes-fixture",
        "akash-fixture",
        "scaleway-fixture",
        "friendli-fixture",
        "clarifai-fixture",
        "meta-model-fixture",
        "parasail-fixture",
        "nscale-fixture",
    ]
    assert all(item.auth_kind == "api_key" for item in candidates)
    assert all(dict(item.config) == {"api_style": "openai"} for item in candidates)
    assert all(item.source.startswith("env:") for item in candidates)
    assert candidates[3].model_name == "custom-nvidia-model"
    assert [item.model_name for item in candidates] == [
        "kimi-k2.6",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.8-27B",
        "custom-nvidia-model",
        "deepseek/deepseek-v3.2",
        "gpt-4o",
        "Meta-Llama-3_3-70B-Instruct",
        "gpt-4o-mini",
        "openai/gpt-4o",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-ai/DeepSeek-V4-Flash",
        "venice-uncensored",
        "Qwen/Qwen3-32B-TEE",
        "zai-org/GLM-5.3",
        "llama-3.3-70b-instruct",
        "zai-org/GLM-5.3-Flash",
        "https://clarifai.com/openai/chat-completion/models/gpt-oss-120b",
        "muse-spark-1.3",
        "parasail-minimax-m3",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
    ]


def test_scan_ignores_blank_values_and_deduplicates_deepinfra_aliases(tmp_path):
    assert scan({"MOONSHOT_API_KEY": "  "}, tmp_path) == []

    candidates = scan(
        {
            "DEEPINFRA_TOKEN": " same-key ",
            "DEEPINFRA_API_KEY": "same-key",
        },
        tmp_path,
    )

    assert len(candidates) == 1
    assert candidates[0].provider == "deepinfra"
    assert candidates[0].source == "env:DEEPINFRA_TOKEN"
    assert candidates[0].secret == "same-key"


def test_scan_keeps_distinct_alias_credentials_and_ignores_blank_override(tmp_path):
    candidates = scan(
        {
            "DEEPINFRA_TOKEN": "token-key",
            "DEEPINFRA_API_KEY": "api-key",
            "TRANCE_MODEL_DEEPINFRA": "  ",
        },
        tmp_path,
    )

    assert [(item.source, item.secret) for item in candidates] == [
        ("env:DEEPINFRA_TOKEN", "token-key"),
        ("env:DEEPINFRA_API_KEY", "api-key"),
    ]
    assert all(item.model_name == "Qwen/Qwen3.8-27B" for item in candidates)

from trance.sources.cloud_api import scan


def test_scan_finds_composite_cloud_credentials_and_models(tmp_path):
    candidates = scan(
        {
            "AZURE_OPENAI_API_KEY": "azure-fixture",
            "AZURE_OPENAI_ENDPOINT": "https://my-resource.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "chat-deployment",
            "CLOUDFLARE_ACCOUNT_ID": "a" * 32,
            "CLOUDFLARE_API_KEY": "cf-fixture",
            "DATABRICKS_TOKEN": "db-fixture",
            "DATABRICKS_HOST": "https://workspace.cloud.databricks.com/",
            "DATABRICKS_MODEL": "serving-model",
            "DASHSCOPE_API_KEY": "dash-fixture",
            "DASHSCOPE_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "TRANCE_MODEL_DASHSCOPE": "qwen-turbo",
        },
        tmp_path,
    )

    assert [(item.provider, item.model_name) for item in candidates] == [
        ("azure-openai", "chat-deployment"),
        ("cloudflare", "@cf/meta/llama-3.1-8b-instruct"),
        ("databricks", "serving-model"),
        ("dashscope", "qwen-turbo"),
    ]
    assert [item.secret for item in candidates] == [
        "azure-fixture",
        "cf-fixture",
        "db-fixture",
        "dash-fixture",
    ]
    assert all(item.auth_kind == "api_key" for item in candidates)
    assert [item.config for item in candidates] == [
        {"base_url": "https://my-resource.openai.azure.com/openai/v1", "api_style": "openai"},
        {
            "base_url": "https://api.cloudflare.com/client/v4/accounts/" + "a" * 32 + "/ai/v1",
            "api_style": "openai",
        },
        {
            "base_url": "https://workspace.cloud.databricks.com/serving-endpoints",
            "api_style": "openai",
        },
        {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "api_style": "openai",
        },
    ]


def test_scan_requires_complete_composite_configuration(tmp_path):
    assert (
        scan(
            {
                "AZURE_OPENAI_API_KEY": "k",
                "AZURE_OPENAI_ENDPOINT": "https://r.openai.azure.com",
            },
            tmp_path,
        )
        == []
    )
    assert scan({"CLOUDFLARE_API_KEY": "k"}, tmp_path) == []
    assert (
        scan(
            {
                "DATABRICKS_TOKEN": "k",
                "DATABRICKS_HOST": "https://x.cloud.databricks.com",
            },
            tmp_path,
        )
        == []
    )


def test_scan_rejects_untrusted_hosts_and_dashscope_coding_plan_endpoint(tmp_path):
    candidates = scan(
        {
            "AZURE_OPENAI_API_KEY": "azure",
            "AZURE_OPENAI_ENDPOINT": "https://attacker.example",
            "AZURE_OPENAI_DEPLOYMENT": "deployment",
            "DATABRICKS_TOKEN": "db",
            "DATABRICKS_HOST": "https://attacker.example",
            "DATABRICKS_MODEL": "model",
            "DASHSCOPE_API_KEY": "dash",
            "DASHSCOPE_BASE_URL": "https://coding.dashscope.aliyuncs.com/v1",
        },
        tmp_path,
    )
    assert candidates == []


def test_scan_rejects_url_injection_and_invalid_cloudflare_account_id(tmp_path):
    candidates = scan(
        {
            "AZURE_OPENAI_API_KEY": "azure",
            "AZURE_OPENAI_ENDPOINT": "https://r.openai.azure.com.attacker.example",
            "AZURE_OPENAI_DEPLOYMENT": "deployment",
            "CLOUDFLARE_ACCOUNT_ID": "../../attacker.example",
            "CLOUDFLARE_API_KEY": "cf",
            "DATABRICKS_TOKEN": "db",
            "DATABRICKS_HOST": "https://workspace.cloud.databricks.com/other/path",
            "DATABRICKS_MODEL": "model",
        },
        tmp_path,
    )
    assert candidates == []


def test_dashscope_uses_documented_default_and_skips_blank_secrets(tmp_path):
    [candidate] = scan({"DASHSCOPE_API_KEY": " dash-fixture "}, tmp_path)
    assert candidate.model_name == "qwen-plus"
    assert candidate.config["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert scan({"DASHSCOPE_API_KEY": " "}, tmp_path) == []

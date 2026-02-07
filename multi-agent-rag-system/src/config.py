"""Application configuration with Pydantic Settings."""

from pydantic_settings import BaseSettings


class AzureOpenAISettings(BaseSettings):
    api_key: str = ""
    endpoint: str = ""
    api_version: str = "2024-08-01-preview"
    chat_deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-large"

    model_config = {"env_prefix": "AZURE_OPENAI_"}


class AzureSearchSettings(BaseSettings):
    endpoint: str = ""
    api_key: str = ""
    index_name: str = "documents"

    model_config = {"env_prefix": "AZURE_SEARCH_"}


class AppSettings(BaseSettings):
    log_level: str = "INFO"
    environment: str = "development"
    cors_origins: list[str] = ["http://localhost:3000"]
    redis_url: str = "redis://localhost:6379/0"

    # RAG parameters
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    rerank_top_n: int = 3
    max_context_tokens: int = 4096

    # Agent parameters
    max_agent_iterations: int = 10
    agent_timeout_seconds: int = 60

    azure_openai: AzureOpenAISettings = AzureOpenAISettings()
    azure_search: AzureSearchSettings = AzureSearchSettings()


settings = AppSettings()

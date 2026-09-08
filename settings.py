import logging
from typing import Any

import sentry_sdk
from decouple import config
from dotenv import load_dotenv
from sentry_sdk.integrations.logging import LoggingIntegration

load_dotenv()


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "True", "yes", "on")
    return bool(value)


def get_envs(env_key: str, cast, default=None) -> str | int | float | bool | Any:
    if cast == bool:
        cast = str_to_bool
    return config(env_key, cast=cast, default=default)


OPENAI_TIMEOUT = get_envs("OPENAI_TIMEOUT", cast=float, default=10.0)
EMBEDDING_BATCH_DELAY = get_envs("EMBEDDING_BATCH_DELAY", cast=float, default=2.0)
ANALYZER_BATCH_DELAY = get_envs("ANALYZER_BATCH_DELAY", cast=float, default=0.01)
OPENAI_API_KEY = get_envs("OPENAI_API_KEY", cast=str, default="")
LLM_MODEL = get_envs("LLM_MODEL", cast=str, default="gpt-4o-mini")
COMPLEMENTARITY_MAX_COMPLETION_TOKENS = get_envs(
    "COMPLEMENTARITY_MAX_COMPLETION_TOKENS", cast=int, default=8000
)
MAX_TOKENS_TUNING = get_envs("MAX_TOKENS_TUNING", cast=int, default=500)
TEMPERATURE = get_envs("TEMPERATURE", cast=int, default=0)

JUDGE_MODEL = get_envs("JUDGE_MODEL", cast=str, default="gpt-4o")
JUDGE_MAX_TOKENS = get_envs("JUDGE_MAX_TOKENS", cast=int, default=2000)

REDIS_URL = get_envs("REDIS_URL", cast=str, default="redis://localhost:6379/0")

MIN_DENSITY = get_envs("MIN_DENSITY", cast=float, default=0.1)
FALLBACK_VALUE = get_envs("FALLBACK_VALUE", cast=float, default=0.5)

DATA_DIR = get_envs("DATA_DIR", cast=str, default="./data")


SENTRY_DSN = get_envs("SENTRY_DSN", cast=str, default="")
SENTRY_ENVIRONMENT = get_envs("SENTRY_ENVIRONMENT", cast=str, default="production")
SENTRY_BREADCRUMB_LEVEL = get_envs("SENTRY_BREADCRUMB_LEVEL", cast=str, default="INFO")
SENTRY_EVENT_LEVEL = get_envs("SENTRY_EVENT_LEVEL", cast=str, default="ERROR")
SENTRY_SEND_PII = get_envs("SENTRY_SEND_PII", cast=bool, default=False)
SENTRY_TRACES_SAMPLE_RATE = get_envs(
    "SENTRY_TRACES_SAMPLE_RATE", cast=float, default=0.0
)
SENTRY_PROFILES_SAMPLE_RATE = get_envs(
    "SENTRY_PROFILES_SAMPLE_RATE", cast=float, default=0.0
)

if SENTRY_DSN:
    breadcrumb_level = getattr(logging, SENTRY_BREADCRUMB_LEVEL.upper(), logging.INFO)
    event_level = getattr(logging, SENTRY_EVENT_LEVEL.upper(), logging.ERROR)

    sentry_logging = LoggingIntegration(
        level=breadcrumb_level,
        event_level=event_level,
    )

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        send_default_pii=SENTRY_SEND_PII,
        environment=SENTRY_ENVIRONMENT,
        auto_enabling_integrations=False,
        integrations=[sentry_logging],
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=SENTRY_PROFILES_SAMPLE_RATE,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(min(root_logger.level or logging.INFO, breadcrumb_level))
    logging.getLogger(__name__).info(
        f"Sentry enabled, reporting {SENTRY_EVENT_LEVEL} and above as events"
    )

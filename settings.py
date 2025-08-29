import logging
from typing import Any

import sentry_sdk
from decouple import config
from dotenv import load_dotenv
from sentry_sdk.integrations.logging import LoggingIntegration

load_dotenv()


def str_to_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "True", "yes", "on")


def get_envs(env_key: str, cast, default=None) -> str | int | float | bool | Any:
    if cast == bool:
        cast = str_to_bool
    return config(env_key, cast=cast, default=default)


OPENAI_TIMEOUT = get_envs("OPENAI_TIMEOUT", cast=float, default=120.0)
EMBEDDING_BATCH_DELAY = get_envs("EMBEDDING_BATCH_DELAY", cast=float, default=2.0)
ANALYZER_BATCH_DELAY = get_envs("ANALYZER_BATCH_DELAY", cast=float, default=2.0)
OPENAI_API_KEY = get_envs("OPENAI_API_KEY", cast=str, default="")
LLM_MODEL = get_envs("LLM_MODEL", cast=str, default="gpt-4o-mini")
MAX_TOKENS = get_envs("MAX_TOKENS", cast=int, default=6000)
MAX_TOKENS_DEDUP = get_envs("MAX_TOKENS_DEDUP", cast=int, default=50)
MAX_TOKENS_TUNING = get_envs("MAX_TOKENS_TUNING", cast=int, default=500)
TEMPERATURE = get_envs("TEMPERATURE", cast=int, default=0)

REDIS_URL = get_envs("REDIS_URL", cast=str, default="redis://localhost:6379/0")

MIN_DENSITY = get_envs("MIN_DENSITY", cast=float, default=0.1)
FALLBACK_VALUE = get_envs("FALLBACK_VALUE", cast=float, default=0.5)

DATA_DIR = get_envs("DATA_DIR", cast=str, default="./data")

SENTRY_DSN = get_envs("SENTRY_DSN", cast=str, default="")
SENTRY_ENVIRONMENT = get_envs("SENTRY_ENVIRONMENT", cast=str, default="production")
SENTRY_LOG_LEVEL = get_envs("SENTRY_LOG_LEVEL", cast=str, default="INFO")

if SENTRY_DSN:
    sentry_logging = LoggingIntegration(
        level=getattr(logging, SENTRY_LOG_LEVEL.upper(), logging.INFO),
        event_level=getattr(logging, SENTRY_LOG_LEVEL.upper(), logging.INFO),
    )

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        send_default_pii=True,
        environment=SENTRY_ENVIRONMENT,
        auto_enabling_integrations=False,
        integrations=[sentry_logging],
        traces_sample_rate=get_envs(
            "SENTRY_TRACES_SAMPLE_RATE", cast=float, default=1.0
        ),
        profiles_sample_rate=get_envs(
            "SENTRY_PROFILES_SAMPLE_RATE", cast=float, default=1.0
        ),
    )

    root_logger = logging.getLogger()

    sentry_handler_exists = any(
        isinstance(handler, type(sentry_logging)) for handler in root_logger.handlers
    )

    if not sentry_handler_exists:
        sentry_level = getattr(logging, SENTRY_LOG_LEVEL.upper())
        root_logger.setLevel(min(root_logger.level or logging.INFO, sentry_level))
    print(f"Sentry logging integration configured - capturing {SENTRY_LOG_LEVEL}+ logs")
else:
    print("Sentry DSN not configured - logs will not be sent to Sentry")

"""Declarative schema of provider config that must be collected before boot.

Detection deliberately runs against raw sources (CLI kwargs, ``os.environ``,
the persisted global config) rather than a resolved ``OpenDataSciConfig``
instance, since several fields (e.g. ``aws_region``) carry a hardcoded
pydantic default that would otherwise mask a value the user never actually
provided.
"""

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.providers import Provider


@dataclass(frozen=True)
class RequiredField:
    """One piece of provider config the onboarding overlay may need to collect."""

    field: str
    label: str
    secret: bool = False
    default: str | None = None
    # If every one of these env vars is set, this field is considered
    # satisfied through an alternate auth path (e.g. Azure service principal)
    # and is skipped entirely.
    skip_if_env: tuple[str, ...] = ()


PROVIDER_REQUIRED_FIELDS: MappingProxyType[Provider, tuple[RequiredField, ...]] = MappingProxyType(
    {
        Provider.ANTHROPIC: (RequiredField("anthropic_api_key", "Anthropic API key", secret=True),),
        Provider.OPENAI: (RequiredField("openai_api_key", "OpenAI API key", secret=True),),
        Provider.GEMINI: (RequiredField("google_api_key", "Google API key", secret=True),),
        Provider.AZURE: (
            RequiredField(
                "azure_api_key",
                "Azure OpenAI API key",
                secret=True,
                skip_if_env=("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"),
            ),
            RequiredField("azure_endpoint", "Azure OpenAI endpoint URL"),
        ),
        Provider.BEDROCK: (RequiredField("aws_region", "AWS region", default="us-east-1"),),
        Provider.VERTEXAI: (
            RequiredField("google_cloud_project", "GCP project ID"),
            RequiredField("google_cloud_location", "Vertex AI location", default="us-central1"),
        ),
        Provider.OLLAMA: (),
        Provider.OPENAI_COMPATIBLE_SERVER: (),
    }
)


def compute_missing_fields(
    providers: Iterable[Provider],
    kwargs: dict[str, object],
    global_cfg: dict[str, object],
) -> list[RequiredField]:
    """Return the required fields not satisfied by *kwargs*, env, or *global_cfg*.

    *providers* may list the primary and secondary provider together; fields
    are deduplicated by name so a shared field (unlikely, but possible) is
    only asked once.
    """
    seen: set[str] = set()
    missing: list[RequiredField] = []
    for provider in providers:
        for rf in PROVIDER_REQUIRED_FIELDS.get(provider, ()):
            if rf.field in seen:
                continue
            seen.add(rf.field)
            if rf.skip_if_env and all(os.environ.get(var) for var in rf.skip_if_env):
                continue
            if kwargs.get(rf.field):
                continue
            model_field = OpenDataSciConfig.model_fields.get(rf.field)
            alias = model_field.alias if model_field is not None else None
            if alias and os.environ.get(alias):
                continue
            if global_cfg.get(rf.field):
                continue
            missing.append(rf)
    return missing

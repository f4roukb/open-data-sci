"""Shared pydantic base classes for the codebase's value objects."""

from pydantic import BaseModel, ConfigDict


class FrozenStrictBaseModel(BaseModel):
    """Base for immutable value objects: frozen, no unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class FrozenBaseModel(BaseModel):
    """Base for immutable LLM structured-output shapes: frozen, unknown fields allowed.

    Unlike :class:`FrozenStrictBaseModel`, this does not set ``extra="forbid"`` — LLM
    providers can emit additional fields outside our control, and rejecting those would
    fail the whole structured-output call over data we don't otherwise use.
    """

    model_config = ConfigDict(frozen=True)


class MutableStrictBaseModel(BaseModel):
    """Base for mutable value objects: assignment allowed, no unknown fields."""

    model_config = ConfigDict(extra="forbid")

"""Shared pydantic base classes for the codebase's value objects."""

from pydantic import BaseModel, ConfigDict


class FrozenStrictBaseModel(BaseModel):
    """Base for immutable value objects: frozen, no unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MutableStrictBaseModel(BaseModel):
    """Base for mutable value objects: assignment allowed, no unknown fields."""

    model_config = ConfigDict(extra="forbid")

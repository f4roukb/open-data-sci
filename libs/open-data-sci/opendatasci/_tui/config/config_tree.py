"""Data model for the selection-driven config panels (/config, /settings, /models).

Pure logic, no Textual — the menu tree and its option lists are plain data so
``config_screen.py`` (rendering) and ``startup_wizard_screen.py`` (the linear
startup flow) can both walk the same structures, and so this module stays
unit-testable without a running app.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

from opendatasci._tui import tips as _tips
from opendatasci._tui.chat.commands import _PROVIDER_DISPLAY
from opendatasci._tui.style import theme as _theme
from opendatasci.configs import (
    DEFAULT_MODEL,
    DEFAULT_SECONDARY_MODEL,
    PRIMARY_INCOMPATIBLE_MODELS,
    OpenDataSciConfig,
)
from opendatasci.models.providers import Provider

# Providers with no fixed model catalog — the model is whatever the user's
# self-hosted endpoint exposes, so picking one always falls back to free text.
_NO_CATALOG_PROVIDERS = frozenset({Provider.OLLAMA, Provider.OPENAI_COMPATIBLE_SERVER})

# The four OpenDataSciConfig fields a startup/config selection can touch.
SELECTION_FIELDS: tuple[str, ...] = ("provider", "model", "secondary_provider", "secondary_model")


@dataclass(frozen=True)
class ConfigOption:
    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ConfigLeaf:
    field: str
    static_options: list[ConfigOption] | None = None
    options_provider: Callable[[dict[str, str]], list[ConfigOption]] | None = None
    text_placeholder: str = ""
    # When this leaf's staged value changes, also reset staged[linked_field]
    # to linked_default(new_value) — used so picking a new provider resets
    # its paired model to that provider's default.
    linked_field: str | None = None
    linked_default: Callable[[str], str] | None = None
    # "mcp_servers" opts this leaf out of the generic choice/text rendering —
    # ConfigScreen instead drives its dedicated add/remove flow for it.
    kind: Literal["choice", "mcp_servers"] = "choice"
    # Whether submitting an empty text value stages "" (meaning "unset / use
    # the default") rather than being treated as a no-op. Only leaves with an
    # explicit null-equals-default meaning (e.g. temperature) set this.
    allow_empty: bool = False
    # Optional validator run against a text leaf's submitted value before
    # staging. Returns an error message to reject the value (the input stays
    # open for another attempt), or None to accept it. Never called for an
    # empty submission when allow_empty is True.
    validate: Callable[[str], str | None] | None = None

    def options(self, staged: dict[str, str]) -> list[ConfigOption]:
        """Selectable options for the current *staged* values.

        An empty result means "no catalog" — the caller should fall back to
        a free-text input instead of a list.
        """
        if self.static_options is not None:
            return self.static_options
        if self.options_provider is not None:
            return self.options_provider(staged)
        return []


@dataclass(frozen=True)
class ConfigNode:
    key: str
    label: str
    children: list["ConfigNode"] = field(default_factory=list)
    leaf: ConfigLeaf | None = None
    # A non-selectable section label rendered inline among its siblings (e.g.
    # "Primary Model" grouping the Model/Temperature rows that follow it) —
    # has no children and no leaf of its own.
    header: bool = False


def _provider_options(_staged: dict[str, str]) -> list[ConfigOption]:
    options = [ConfigOption(p.value, _PROVIDER_DISPLAY.get(p, p.value.title())) for p in Provider]
    return sorted(options, key=lambda o: o.label.lower())


def _model_options_for(
    provider_field: str, role: Literal["primary", "secondary"]
) -> Callable[[dict[str, str]], list[ConfigOption]]:
    def resolve(staged: dict[str, str]) -> list[ConfigOption]:
        try:
            provider = Provider(staged.get(provider_field, ""))
        except ValueError:
            return []
        if provider in _NO_CATALOG_PROVIDERS:
            return []
        candidates = {DEFAULT_MODEL.get(provider), DEFAULT_SECONDARY_MODEL.get(provider)}
        if role == "primary":
            candidates -= PRIMARY_INCOMPATIBLE_MODELS.get(provider, frozenset())
        return [ConfigOption(m, m) for m in sorted(c for c in candidates if c)]

    return resolve


def _default_model_for(provider_value: str) -> str:
    try:
        provider = Provider(provider_value)
    except ValueError:
        return ""
    return DEFAULT_MODEL.get(provider, "")


def _default_secondary_model_for(provider_value: str) -> str:
    try:
        provider = Provider(provider_value)
    except ValueError:
        return ""
    return DEFAULT_SECONDARY_MODEL.get(provider, "")


def build_theme_leaf() -> ConfigLeaf:
    return ConfigLeaf(
        field="theme",
        static_options=[
            ConfigOption(name, name, desc) for name, desc in _theme.THEME_DESCRIPTIONS.items()
        ],
    )


def build_tips_leaf() -> ConfigLeaf:
    return ConfigLeaf(
        field="tips",
        static_options=[
            ConfigOption("on", "On", "Show rotating tips in the footer"),
            ConfigOption("off", "Off", "Hide the footer tips"),
        ],
    )


def build_provider_leaf(field_name: str, linked_field: str) -> ConfigLeaf:
    linked_default = _default_model_for if linked_field == "model" else _default_secondary_model_for
    return ConfigLeaf(
        field=field_name,
        options_provider=_provider_options,
        linked_field=linked_field,
        linked_default=linked_default,
    )


def build_model_leaf(
    field_name: str, provider_field: str, role: Literal["primary", "secondary"]
) -> ConfigLeaf:
    return ConfigLeaf(
        field=field_name,
        options_provider=_model_options_for(provider_field, role),
        text_placeholder="Model name",
    )


def build_mcp_servers_leaf() -> ConfigLeaf:
    return ConfigLeaf(field="mcp_servers", kind="mcp_servers")


def build_skills_leaf() -> ConfigLeaf:
    return ConfigLeaf(field="skills_directory", text_placeholder="Path to skills folder")


def _validate_temperature(value: str) -> str | None:
    try:
        parsed = float(value)
    except ValueError:
        return "Temperature must be a number between 0.0 and 1.0"
    if not 0.0 <= parsed <= 1.0:
        return "Temperature must be between 0.0 and 1.0"
    return None


def build_temperature_leaf() -> ConfigLeaf:
    return ConfigLeaf(
        field="primary_temperature",
        text_placeholder="0.0-1.0 (leave empty for default)",
        allow_empty=True,
        validate=_validate_temperature,
    )


def build_agent_name_leaf() -> ConfigLeaf:
    return ConfigLeaf(field="name", text_placeholder="Agent name")


def build_worker_timeout_leaf() -> ConfigLeaf:
    return ConfigLeaf(field="worker_timeout_seconds", text_placeholder="Seconds")


def _format_number(value: float) -> str:
    """Render a float without a trailing ``.0`` for whole numbers."""
    return str(int(value)) if value == int(value) else str(value)


def build_config_tree() -> ConfigNode:
    """The full /config (alias /settings) menu. Sections, and entries within
    each section, are kept in lexical (alphabetical) order, except within
    "Models": provider precedes model there since the model catalog offered
    depends on whichever provider is currently staged."""
    return ConfigNode(
        key="root",
        label="Configure",
        children=[
            ConfigNode(
                key="display",
                label="Display",
                children=[
                    ConfigNode(key="theme", label="Theme", leaf=build_theme_leaf()),
                    ConfigNode(key="tips", label="Tips", leaf=build_tips_leaf()),
                ],
            ),
            ConfigNode(
                key="integrations",
                label="Integrations",
                children=[
                    ConfigNode(
                        key="mcp_servers", label="MCP Servers", leaf=build_mcp_servers_leaf()
                    ),
                    ConfigNode(
                        key="skills_directory",
                        label="Skills directory",
                        leaf=build_skills_leaf(),
                    ),
                ],
            ),
            ConfigNode(
                key="models",
                label="Models",
                children=[
                    ConfigNode(key="primary_model_header", label="Primary Model", header=True),
                    ConfigNode(
                        key="primary_provider",
                        label="Provider",
                        leaf=build_provider_leaf("provider", "model"),
                    ),
                    ConfigNode(
                        key="primary_model",
                        label="Model",
                        leaf=build_model_leaf("model", "provider", "primary"),
                    ),
                    ConfigNode(
                        key="primary_temperature",
                        label="Temperature",
                        leaf=build_temperature_leaf(),
                    ),
                    ConfigNode(key="secondary_model_header", label="Secondary Model", header=True),
                    ConfigNode(
                        key="secondary_provider",
                        label="Provider",
                        leaf=build_provider_leaf("secondary_provider", "secondary_model"),
                    ),
                    ConfigNode(
                        key="secondary_model",
                        label="Model",
                        leaf=build_model_leaf("secondary_model", "secondary_provider", "secondary"),
                    ),
                ],
            ),
            ConfigNode(
                key="personalization",
                label="Personalization",
                children=[
                    ConfigNode(key="agent_name", label="Agent name", leaf=build_agent_name_leaf()),
                ],
            ),
            ConfigNode(
                key="subagents",
                label="Subagents",
                children=[
                    ConfigNode(
                        key="worker_timeout",
                        label="Worker timeout",
                        leaf=build_worker_timeout_leaf(),
                    ),
                ],
            ),
        ],
    )


def initial_values(cfg: OpenDataSciConfig, theme_name: str) -> dict[str, str]:
    """Seed the staged-values dict from the current config and active theme."""
    return {
        "theme": theme_name,
        "tips": "on" if _tips.enabled else "off",
        "provider": str(cfg.provider),
        "model": cfg.model,
        "secondary_provider": str(cfg.secondary_provider),
        "secondary_model": cfg.secondary_model,
        "skills_directory": str(cfg.skills_directory) if cfg.skills_directory else "",
        "primary_temperature": (
            "" if cfg.primary_temperature == 0.0 else _format_number(cfg.primary_temperature)
        ),
        "name": cfg.name,
        "worker_timeout_seconds": (
            "" if cfg.worker_timeout_seconds is None else _format_number(cfg.worker_timeout_seconds)
        ),
    }


def diff_values(initial: dict[str, str], staged: dict[str, str]) -> dict[str, str]:
    """Only the keys whose value actually changed."""
    return {k: v for k, v in staged.items() if initial.get(k) != v}

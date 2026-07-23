"""Slash command registry and display-text formatters for the OpenDataSci TUI."""

import re

from opendatasci.models.providers import Provider

SLASH_COMMANDS: list[str] = [
    "/cancel-all-messages",
    "/cancel-message",
    "/clear",
    "/compact",
    "/exit",
    "/help",
    "/ls-workspace",
    "/model",
    "/models",
    "/provider",
    "/reset",
    "/secondary-model",
    "/secondary-provider",
    "/stop",
    "/theme",
    "/themes",
]

SLASH_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/cancel-all-messages": "Cancel all queued messages",
    "/cancel-message": "Cancel the most recently queued message",
    "/clear": "Clear conversation context",
    "/compact": "Summarize conversation history",
    "/exit": "Exit OpenDataSci",
    "/help": "Show all commands",
    "/ls-workspace": "List workspace files",
    "/model": "Switch the primary model, e.g. /model claude-opus-4-8",
    "/models": "Show models in use",
    "/provider": "Switch provider (and model), e.g. /provider openai",
    "/reset": "Reset agent session",
    "/secondary-model": "Switch the secondary model, e.g. /secondary-model gpt-5.6-luna",
    "/secondary-provider": "Switch secondary provider (and model), e.g. /secondary-provider openai",
    "/stop": "Stop the running agent",
    "/theme": "Switch colour theme instantly, e.g. /theme dracula",
    "/themes": "List available colour themes",
}

_PROVIDER_DISPLAY: dict[Provider, str] = {
    Provider.ANTHROPIC: "Anthropic",
    Provider.OPENAI: "OpenAI",
    Provider.BEDROCK: "AWS Bedrock",
    Provider.GEMINI: "Google",
    Provider.VERTEXAI: "Google Vertex AI",
    Provider.AZURE: "Azure OpenAI",
    Provider.OLLAMA: "Ollama",
    Provider.OPENAI_COMPATIBLE_SERVER: "OpenAI-compatible server",
}


def _fmt_model(provider: str, model_id: str) -> str:
    try:
        provider_label = _PROVIDER_DISPLAY[Provider(provider)]
    except (KeyError, ValueError):
        provider_label = provider.title()
    m = re.search(r"claude-([a-z]+)-(\d+)(?:-(\d+))?", model_id)
    if m:
        variant, major, minor = m.groups()
        version = f"{major}.{minor}" if minor else major
        return f"{provider_label} Claude {variant.title()} {version}"
    return f"{provider_label} {model_id}"


def format_models_message(
    primary_provider: str, model: str, secondary_provider: str, secondary_model: str
) -> str:
    """Return the Markdown text shown by the /models command."""
    lines = [
        "## Models\n",
        f"- **Primary Model**   : {_fmt_model(primary_provider, model)}",
        f"- **Secondary Model** : {_fmt_model(secondary_provider, secondary_model)}",
    ]
    return "\n".join(lines)


def format_help_message() -> str:
    """Return the Markdown text shown by the /help command."""
    lines = [
        "## Available Commands\n",
        "- **/cancel-all-messages** — Cancel all messages queued while the agent was busy",
        "- **/cancel-message** — Cancel the most recently queued message",
        "- **/clear** — Clear all conversation context, including the plan (preserves session variables)",
        "- **/compact** — Summarize and compress the conversation history",
        "- **/exit** — Exit OpenDataSci",
        "- **/help** — Show this help message",
        "- **/ls-workspace** — List files in the workspace",
        "- **/model \\<name\\>** — Switch the primary model, e.g. `/model claude-opus-4-8`",
        "- **/models** — Show the primary and secondary model in use",
        "- **/provider \\<name\\> [model]** — Switch provider (and optionally model), "
        "e.g. `/provider openai`",
        "- **/reset** — Reset the agent session and reload data from disk",
        "- **/secondary-model \\<name\\>** — Switch the secondary model, "
        "e.g. `/secondary-model gpt-5.6-luna`",
        "- **/secondary-provider \\<name\\> [model]** — Switch the secondary provider "
        "(and optionally model), e.g. `/secondary-provider openai`",
        "- **/stop** — Stop the running agent (future messages pick up where it left off)",
        "- **/theme \\<name\\>** — Switch colour theme instantly, e.g. `/theme dracula`",
        "- **/themes** — List available colour themes",
    ]
    lines.append("\n**Tip:** Type `/` to see commands via autocomplete, or `@` to attach a file.")
    return "\n".join(lines)


def format_themes_message(active_name: str, themes: dict[str, str]) -> str:
    """Return the Markdown text shown by the /themes command.

    `themes` is a mapping of theme name to a one-line description.
    """
    lines = ["## Colour Themes\n"]
    for name, description in themes.items():
        marker = " *(active)*" if name == active_name else ""
        lines.append(f"- **{name}**{marker} — {description}")
    lines.append("\nSwitch themes by relaunching with `--theme <name>`.")
    return "\n".join(lines)


def format_theme_switched_message(name: str) -> str:
    """Return the Markdown text shown after a successful `/theme <name>` switch."""
    return f"✓ Theme switched to **{name}**."


def format_unknown_theme_message(name: str, themes: dict[str, str]) -> str:
    """Return the Markdown text shown when `/theme <name>` names an unknown theme."""
    valid = ", ".join(f"`{theme_name}`" for theme_name in themes)
    return f"✗ Unknown theme: `{name}`\n\nValid themes: {valid}"


def format_model_switched_message(provider: str, model: str) -> str:
    """Return the Markdown text shown after a successful model/provider switch."""
    return f"✓ Now using {_fmt_model(provider, model)}."


def format_unknown_provider_message(name: str) -> str:
    """Return the Markdown text shown when `/provider <name>` names an unknown provider."""
    valid = ", ".join(p.value for p in Provider)
    return f"✗ Unknown provider: `{name}`\n\nValid providers: {valid}"


def format_missing_api_key_message(provider: str, key_field: str) -> str:
    """Return the Markdown text shown when switching to *provider* needs a missing API key."""
    try:
        provider_label = _PROVIDER_DISPLAY[Provider(provider)]
    except (KeyError, ValueError):
        provider_label = provider.title()
    env_var = key_field.upper()
    return (
        f"✗ No API key configured for **{provider_label}**.\n\n"
        f"Set the `{env_var}` environment variable and restart, "
        f"or relaunch with `--provider {provider} --api-key <key>`."
    )

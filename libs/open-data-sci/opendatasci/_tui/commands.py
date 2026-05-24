"""Slash command registry and display-text formatters for the OpenDataSci TUI."""

import re

from opendatasci.models.providers import Provider

SLASH_COMMANDS: list[str] = [
    "/clear",
    "/compact",
    "/exit",
    "/help",
    "/ls-workspace",
    "/models",
    "/reset",
    "/stop",
    "/themes",
]

SLASH_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/clear": "Clear conversation context",
    "/compact": "Summarize conversation history",
    "/exit": "Exit OpenDataSci",
    "/help": "Show all commands",
    "/ls-workspace": "List workspace files",
    "/models": "Show models in use",
    "/reset": "Reset agent session",
    "/stop": "Stop the running agent",
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
    Provider.VLLM: "vLLM",
}


def _fmt_model(provider: str, model_id: str) -> str:
    try:
        provider_label = _PROVIDER_DISPLAY[Provider(provider)]
    except (KeyError, ValueError):
        provider_label = provider.title()
    m = re.search(r"claude-([a-z]+)-(\d+)-(\d+)", model_id)
    if m:
        variant, major, minor = m.groups()
        return f"{provider_label} Claude {variant.title()} {major}.{minor}"
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
        "- **/clear** — Clear the conversation (preserves session variables)",
        "- **/compact** — Summarize and compress the conversation history",
        "- **/exit** — Exit OpenDataSci",
        "- **/help** — Show this help message",
        "- **/ls-workspace** — List files in the workspace",
        "- **/models** — Show the primary and secondary model in use",
        "- **/reset** — Reset the agent session and reload data from disk",
        "- **/stop** — Stop the running agent (future messages pick up where it left off)",
        "- **/themes** — List available colour themes (selected at launch with `--theme`)",
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

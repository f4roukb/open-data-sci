"""Slash command registry and display-text formatters for the OpenDataSci TUI."""

from opendatasci.models.providers import Provider

SLASH_COMMANDS: list[str] = [
    "/cancel-all-messages",
    "/cancel-message",
    "/clear",
    "/compact",
    "/config",
    "/exit",
    "/help",
    "/ls-workspace",
    "/models",
    "/providers",
    "/reset",
]

SLASH_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/cancel-all-messages": "Cancel all queued messages",
    "/cancel-message": "Cancel the most recently queued message",
    "/clear": "Clear conversation context",
    "/compact": "Summarize conversation history",
    "/config": "Open the configuration panel (display, models, providers)",
    "/exit": "Exit OpenDataSci",
    "/help": "Show all commands",
    "/ls-workspace": "List workspace files",
    "/models": "Pick the primary and secondary model",
    "/providers": "Pick the primary and secondary provider",
    "/reset": "Reset agent session",
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


def format_help_message() -> str:
    """Return the Markdown text shown by the /help command."""
    lines = [
        "## Available Commands\n",
        "- **/cancel-all-messages** — Cancel all messages queued while the agent was busy",
        "- **/cancel-message** — Cancel the most recently queued message",
        "- **/clear** — Clear all conversation context",
        "- **/compact** — Summarize and compress the conversation history",
        "- **/config** — Open the configuration panel (display, models, providers)",
        "- **/exit** — Exit OpenDataSci",
        "- **/help** — Show this help message",
        "- **/ls-workspace** — List files in the workspace",
        "- **/models** — Pick the primary and secondary model",
        "- **/providers** — Pick the primary and secondary provider",
        "- **/reset** — Reset the agent session and reload data from disk",
    ]
    lines.append("\n**Tip:** Type `/` to see commands via autocomplete, or `@` to attach a file.")
    return "\n".join(lines)


def format_missing_api_key_message(provider: str, key_field: str) -> str:
    """Return the Markdown text shown when switching to *provider* needs a missing API key."""
    try:
        provider_label = _PROVIDER_DISPLAY[Provider(provider)]
    except (KeyError, ValueError):
        provider_label = provider.title()
    env_var = key_field.upper()
    return (
        f"No API key configured for {provider_label}. "
        f"Set the {env_var} environment variable and try again."
    )

_CONNECTION_EXC_NAMES = frozenset(
    ["ConnectError", "APIConnectionError", "ConnectionError", "NetworkError"]
)
_CONNECTION_KEYWORDS = frozenset(
    ["connection error", "connection refused", "name or service not known"]
)


def format_stream_error(exc: Exception) -> str:
    """Return a user-friendly error message for a streaming exception."""
    exc_type = type(exc).__name__
    msg = str(exc)
    msg_lower = msg.lower()
    if exc_type in _CONNECTION_EXC_NAMES or any(kw in msg_lower for kw in _CONNECTION_KEYWORDS):
        return (
            "Connection error — could not reach the API. "
            "Check your internet connection and verify your API key is valid."
        )
    if exc_type in ("AuthenticationError", "AuthError") or any(
        kw in msg_lower for kw in ("authentication", "api_key", "invalid x-api-key", "unauthorized")
    ):
        return f"Authentication error — your API key may be invalid or expired. ({msg})"
    return msg

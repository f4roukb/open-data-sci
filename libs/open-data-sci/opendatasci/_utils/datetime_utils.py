from datetime import datetime, timezone


def datetime_now() -> datetime:
    return datetime.now(timezone.utc)


def to_local_timezone(dt: datetime) -> datetime:
    """Convert a UTC-aware datetime to the OS local timezone."""
    return dt.astimezone()

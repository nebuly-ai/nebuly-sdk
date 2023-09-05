class NebulyException(Exception):
    """Base class for exceptions in this module."""

    pass


class NebulyApiKeyRequired(NebulyException):
    """Exception raised when NEBULY_API_KEY is not set."""

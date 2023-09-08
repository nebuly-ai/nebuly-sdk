from __future__ import annotations


class NebulyException(Exception):
    """Base class for exceptions in this module."""


class NebulyHTTPError(NebulyException):
    """Exception raised for errors in the HTTP request."""


class AlreadyImportedError(NebulyException):
    """Exception raised when the ai package was imported before init."""


class NebulyAlreadyInitializedError(NebulyException):
    """Exception raised when init is called more than once."""


class APIKeyNotProvidedError(NebulyException):
    """Exception raised when the API key is not provided."""

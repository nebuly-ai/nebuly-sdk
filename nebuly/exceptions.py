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


class InvalidNebulyKeyError(NebulyException):
    """Exception raised when the API key is not valid."""


class MissingRequiredNebulyFieldError(NebulyException):
    """Exception raised when a required nebuly field is missing."""


class InteractionContextError(NebulyException):
    """Exception raised when the API key is not valid."""


class InteractionContextInitiationError(InteractionContextError):
    """Exception raised when the interaction context is not initialized properly."""


class AlreadyInInteractionContext(InteractionContextError):
    """Exception raised when one interaction context is initialized inside another"""


class NotInInteractionContext(InteractionContextError):
    """Exception raised when no interaction context can be found."""


class InteractionMustBeLocalVariable(InteractionContextError):
    """Exception raised when the interaction manager creation misses the 'as'"""

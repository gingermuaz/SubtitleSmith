"""Errors shared by engine.py, writers.py and burn.py."""


class ProcessingError(Exception):
    """A problem with one file, worded for the user."""


class ProcessCancelled(Exception):
    """The user pressed Cancel."""

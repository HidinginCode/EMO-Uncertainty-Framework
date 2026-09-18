"""This module contains a logger that adapts to other classes."""

import logging

class LoggingMixin:
    """This class creates a logger if inherited from."""
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.logger = logging.getLogger(f"{cls.__qualname__}")
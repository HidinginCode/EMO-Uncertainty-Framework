"""This module contains the logger setup."""

# System imports
import os
import sys

# Python imports
import logging
import logging.config
import shutil

_configured = False

def setup_logging(level=logging.INFO):

    _LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "python_logs")
    os.makedirs(_LOG_DIR, exist_ok=True)

    # Sanity check if configured by other instance
    global _configured
    if _configured:
        return

    logging.config.dictConfig({
        "version": 1.0,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "[%(asctime)s][%(levelname)s][%(name)s]: %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": os.path.join(_LOG_DIR, "full.log"),
                "mode": "a"
            }
        },
        "root": {"handlers": ["console", "file"], "level": level},
    })
    _configured = True

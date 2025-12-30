"""Logger configuration module."""

import logging
import sys
from typing import Optional


def getLogger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance with formatted output.
    
    Args:
        name: The name of the logger (typically __name__ of the calling module)
        level: Optional log level (defaults to INFO if not specified)
    
    Returns:
        A configured Logger instance with formatted output structure
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times if logger already configured
    if logger.handlers:
        return logger
    
    # Set log level
    if level is None:
        level = logging.INFO
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter with structured output
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger


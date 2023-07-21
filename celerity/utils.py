"""
Some general utility functions.
"""
import logging
import sys


def get_logger(logger_name: str, file_name: str = None,
               format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> logging.Logger:
    """
    Returns a logger.

    Parameters
    ----------
    logger_name: the name of logger (should be the name of the module that calls this function)
    file_name: optional. If not None then an additional logger will be created that writes to a file.
    format: optional. The format of the logging prefix.

    Returns
    -------
    a logging.Logger

    Examples
    -------
    > logger = get_logger(__name__)
    """
    # Create handlers
    logger = logging.Logger(logger_name)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter(format)
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    if file_name is not None:
        f_handler = logging.FileHandler(file_name)
        f_handler.setLevel(logging.ERROR)

        # Create formatters and add it to handlers
        f_format = logging.Formatter(format)
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    logger.propagate = False
    return logger
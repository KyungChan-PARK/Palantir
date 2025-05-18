import logging
from typing import Optional

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    level: int = logging.INFO, log_file: Optional[str] = None
) -> None:
    """Configure global logging.

    Parameters
    ----------
    level:
        Logging level for the root logger.
    log_file:
        Optional path to a file where logs should also be written.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)

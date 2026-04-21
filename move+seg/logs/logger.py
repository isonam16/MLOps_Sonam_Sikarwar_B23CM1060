import logging
import os

def get_logger(name: str = "app_logger"):
    """
    Returns a configured logger instance.
    Prevents duplicate handlers if called multiple times.
    Logs to console and (optionally) to a file.
    """

    logger = logging.getLogger(name)

    # If handlers already exist, just return the logger
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional: write logs to file under logs/app.log
    log_dir = "logs"
    log_file = os.path.join(log_dir, "app.log")
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If writing to file fails (e.g., read-only FS), ignore silently
        pass

    return logger

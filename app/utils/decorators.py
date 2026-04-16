import time
import logging
import functools
from typing import Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("system_logger")

def log_execution(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"🚀 Executing {func.__name__}...")
        result = func(*args, **kwargs)
        logger.info(f"✅ Completed {func.__name__}")
        return result
    return wrapper

def timing(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"⏱️ {func.__name__} took {duration:.4f} seconds")
        return result, duration
    return wrapper
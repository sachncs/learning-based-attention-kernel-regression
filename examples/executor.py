"""Example executor providing structured logging and timing for demonstrations.

This module implements :class:`ExampleExecutor`, a concrete sub-class
of :class:`~laker.executor.Executor` designed for end-to-end example
scripts.  It provides:

* **Section headers** rendered as ``=``-delimited banners.
* **Named result logging** stored in an internal dictionary for
  downstream access.
* **Timed operations** that measure wall-clock time and log the result
  in seconds.
* **Formatted metric logging** for numeric values.

Usage::

    executor = ExampleExecutor("My Example")
    executor.section("Step 1")
    executor.log_result("accuracy", 0.95)
    executor.time_operation("fit", model.fit, X, y)
"""

import logging
import time
from typing import Any, Callable, Dict

from laker.executor import Executor

logger = logging.getLogger(__name__)


class ExampleExecutor(Executor):
    """Executes example workflows with structured logging and timing.

    A lightweight executor that stores results in an in-memory
    dictionary and logs all output at ``INFO`` level.  Suitable for
    Jupyter notebooks, scripts, and documentation examples where
    statistical aggregation is not required.

    Args:
        name: Human-readable label for this executor instance, used in
            log messages.

    Attributes:
        results: Dictionary of named results accumulated during the
            example run.
    """

    def __init__(self, name: str = "LAKER Example"):
        self.name = name
        self.results: Dict[str, Any] = {}

    def section(self, title: str) -> None:
        """Log a section header.

        Emits a ``=``-delimited banner at ``INFO`` level to visually
        separate phases of an example workflow.

        Args:
            title: Human-readable section title.
        """
        logger.info("=" * 60)
        logger.info(title)
        logger.info("=" * 60)

    def log_result(self, key: str, value: Any) -> None:
        """Log a named result and store it internally.

        The result is both logged at ``INFO`` level and stored in
        :attr:`results` for later retrieval.

        Args:
            key: Human-readable result label.  Should be unique within
                the current section.
            value: Result value (any JSON-serialisable type).
        """
        self.results[key] = value
        logger.info("%s: %s", key, value)

    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any:
        """Run an operation and log its elapsed time.

        Measures wall-clock time around ``operation()`` and logs the
        result in seconds.  The operation's return value is propagated
        unchanged.

        Args:
            name: Human-readable operation name.
            operation: Zero-argument callable to execute.

        Returns:
            Whatever ``operation`` returns.
        """
        logger.info("Starting: %s", name)
        start = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - start
        logger.info("Completed: %s in %.3fs", name, elapsed)
        return result

    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        """Log a numeric metric with formatting.

        The metric is both logged at ``INFO`` level and stored in
        :attr:`results` for later retrieval.

        Args:
            name: Metric identifier.  Should be unique within the
                current section.
            value: Numeric value.
            fmt: Python format specification applied to ``value`` for
                display.  Defaults to four decimal places.
        """
        formatted = format(value, fmt)
        self.results[name] = value
        logger.info("%s: %s", name, formatted)

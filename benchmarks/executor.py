"""Benchmark executor providing standardised timing and logging.

This module implements :class:`BenchmarkExecutor`, a concrete
sub-class of :class:`~laker.executor.Executor` designed for
performance measurement.  It provides:

* **Repeated execution** with configurable warmup and trial counts.
* **Single-trial timing** for operations where variance is negligible.
* **Blocked repeated timing** that runs multiple iterations inside a
  single ``time.perf_counter`` window to reduce overhead.
* **Result accumulation** across multiple benchmarks for aggregate
  analysis.

All timing results are reported in milliseconds and returned as
dictionaries with a standardised schema (``mean_ms``, ``std_ms``,
``min_ms``, ``max_ms``, ``trials``).

Usage::

    executor = BenchmarkExecutor(warmup=20, trials=50)
    result = executor.run("my_bench", my_operation)
    print(result["mean_ms"])
"""

import logging
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

from laker.executor import Executor

logger = logging.getLogger(__name__)


class BenchmarkExecutor(Executor):
    """Executes benchmarks with standardised timing, warmup, and logging.

    Extends :class:`~laker.executor.Executor` with statistical timing
    primitives optimised for micro-benchmarking.  Each ``run`` variant
    returns a dictionary with the same schema so that results can be
    trivially aggregated or serialised.

    Args:
        warmup: Default number of warm-up iterations executed before
            measurement in :meth:`run`.
        trials: Default number of measured iterations in :meth:`run`.

    Attributes:
        results: Accumulated list of result dictionaries from all
            benchmark runs executed by this instance.
    """

    def __init__(self, warmup: int = 20, trials: int = 50):
        self.warmup = warmup
        self.trials = trials
        self.results: List[Dict[str, Any]] = []

    def run(
        self,
        name: str,
        operation: Callable[[], Any],
        trials: Optional[int] = None,
        warmup: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run a benchmark operation with warmup and multiple trials.

        Executes ``warmup`` un-timed iterations followed by ``trials``
        individually-timed iterations.  Wall-clock time is measured with
        :func:`time.perf_counter` and recorded in milliseconds.

        Args:
            name: Identifier for this benchmark, used in log messages
                and stored in the result dictionary.
            operation: Zero-argument callable to benchmark.
            trials: Number of measurement iterations.  Defaults to
                ``self.trials`` when ``None``.
            warmup: Number of warm-up iterations.  Defaults to
                ``self.warmup`` when ``None``.

        Returns:
            Dictionary with keys:

            * ``name`` — the benchmark identifier.
            * ``mean_ms`` — arithmetic mean of trial times.
            * ``std_ms`` — sample standard deviation of trial times.
            * ``min_ms`` — minimum trial time.
            * ``max_ms`` — maximum trial time.
            * ``trials`` — number of measured iterations.
        """
        trials = trials if trials is not None else self.trials
        warmup = warmup if warmup is not None else self.warmup

        logger.info("Starting benchmark '%s' (%d warmup, %d trials)", name, warmup, trials)

        for i in range(warmup):
            operation()

        times = []
        for i in range(trials):
            start = time.perf_counter()
            operation()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0

        result = {
            "name": name,
            "mean_ms": mean,
            "std_ms": std,
            "min_ms": min(times),
            "max_ms": max(times),
            "trials": trials,
        }

        self.results.append(result)
        logger.info("Benchmark '%s' completed: mean=%.3fms std=%.3fms", name, mean, std)

        return result

    def run_once(self, name: str, operation: Callable[[], Any]) -> Dict[str, float]:
        """Run a benchmark operation once and record the time.

        Suitable for operations where variance between calls is
        negligible (e.g. large preconditioner builds or full model fits)
        and repeated execution would be prohibitively expensive.

        Args:
            name: Identifier for this benchmark, used in log messages
                and stored in the result dictionary.
            operation: Zero-argument callable to benchmark.

        Returns:
            Dictionary with the standardised schema.  ``std_ms`` is
            ``0.0`` and ``trials`` is ``1``.
        """
        logger.info("Starting single-run benchmark '%s'", name)

        start = time.perf_counter()
        operation()
        elapsed = (time.perf_counter() - start) * 1000

        result = {
            "name": name,
            "mean_ms": elapsed,
            "std_ms": 0.0,
            "min_ms": elapsed,
            "max_ms": elapsed,
            "trials": 1,
        }

        self.results.append(result)
        logger.info("Benchmark '%s' completed: %.3fms", name, elapsed)

        return result

    def section(self, title: str) -> None:
        """Log a section header.

        Emits a ``=``-delimited banner at ``INFO`` level to visually
        separate groups of benchmark results.

        Args:
            title: Human-readable section title.
        """
        logger.info("=" * 60)
        logger.info(title)
        logger.info("=" * 60)

    def log_result(self, key: str, value: Any) -> None:
        """Log a named result.

        Args:
            key: Human-readable label for the result.
            value: Result value (any JSON-serialisable type).
        """
        logger.info("%s: %s", key, value)

    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        """Log a numeric metric with formatting.

        Args:
            name: Metric identifier.
            value: Numeric value to log.
            fmt: Python format specification applied to ``value`` for
                display.  Defaults to four decimal places.
        """
        formatted = format(value, fmt)
        logger.info("%s: %s", name, formatted)

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

    def run_repeated(
        self, name: str, operation: Callable[[], Any], repetitions: int = 20
    ) -> Dict[str, float]:
        """Run an operation multiple times in a single timed block and average.

        Unlike :meth:`run`, all repetitions are executed inside one
        ``time.perf_counter`` window, and the total elapsed time is
        divided by ``repetitions`` to obtain a per-iteration mean.
        This approach reduces per-call measurement overhead and is
        preferred for very fast operations (e.g. kernel matvecs).

        Args:
            name: Identifier for this benchmark, used in log messages
                and stored in the result dictionary.
            operation: Zero-argument callable to benchmark.
            repetitions: Number of repetitions within the single timed
                block.

        Returns:
            Dictionary with the standardised schema.  ``mean_ms`` is
            the per-repetition average, ``std_ms`` is ``0.0`` (no
            per-trial variance is captured), and ``trials`` is set to
            ``repetitions``.
        """
        logger.info("Starting repeated benchmark '%s' (%d repetitions)", name, repetitions)

        start = time.perf_counter()
        for i in range(repetitions):
            operation()
        elapsed = (time.perf_counter() - start) / repetitions * 1000

        result = {
            "name": name,
            "mean_ms": elapsed,
            "std_ms": 0.0,
            "min_ms": elapsed,
            "max_ms": elapsed,
            "trials": repetitions,
        }

        self.results.append(result)
        logger.info("Benchmark '%s' completed: %.3fms per iteration", name, elapsed)

        return result

"""Abstract :class:`Executor` base class for structured logging and timing.

This module defines the single interface that workflow classes
(examples, benchmarks, end-to-end demos) must accept to emit structured
output. The interface intentionally contains only four methods so that
implementations can target a wide range of backends (rich console output,
JSON file logs, markdown reports, etc.).

The pattern is described in detail in ``docs/patterns.md``. The
high-level rules are:

1. Every workflow class accepts ``executor: Optional[Executor] = None``
   in its constructor and falls back to a sensible default
   implementation.
2. Workflows delegate all logging and timing to the executor; library
   code never calls :func:`print` (see ``docs/patterns.md`` section 4).
3. Implementations are interchangeable: the same workflow can be run in
   a notebook, in CI, or in a benchmark suite without modification.

Two concrete implementations are provided by the LAKER package:

- :class:`examples.executor.ExampleExecutor` — structured logging for
  end-to-end demos.
- :class:`benchmarks.executor.BenchmarkExecutor` — timing, warmup, and
  statistical aggregation for performance measurement.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable


class Executor(ABC):
    """Abstract base class for workflow executors.

    An executor provides structured logging, timing, and result
    collection for runnable workflows. Subclasses implement concrete
    logging backends (e.g. console output, benchmark metrics, file
    output).

    Workflow classes should accept an optional :class:`Executor`
    instance in their constructor and delegate all logging and timing
    to it. This keeps business logic decoupled from output formatting
    and makes workflows easy to run from a notebook, a benchmark suite,
    or a CI job without code changes.

    The :class:`Executor` API is intentionally minimal — only four
    methods — so concrete implementations can target arbitrary output
    backends. Implementations may add additional non-abstract helpers
    (e.g. context-manager APIs); these are not part of the contract.

    Thread-safety:
        The base class imposes no thread-safety requirements.
        Implementations that wish to be called from multiple threads
        must provide their own synchronisation.

    See Also:
        :class:`examples.executor.ExampleExecutor`,
        :class:`benchmarks.executor.BenchmarkExecutor`.

    """

    @abstractmethod
    def section(self, title: str) -> None:
        """Log a section header.

        Section headers are typically rendered as horizontal banners
        (e.g. a Markdown ``##`` heading) that visually delimit logical
        phases of a workflow run.

        Args:
            title: Human-readable section title.

        """

    @abstractmethod
    def log_result(self, key: str, value: Any) -> None:
        """Log a named result.

        Results are typically rendered as ``key: value`` pairs (e.g. a
        ``**kwargs``-style dictionary dump or a markdown table row).
        Implementations are expected to handle arbitrary JSON-friendly
        types: ``int``, ``float``, ``str``, ``bool``, ``list``,
        ``dict``, and ``None``.

        Args:
            key: Result identifier. Should be unique within a section.
            value: Result value (any JSON-serialisable type).

        """

    @abstractmethod
    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        """Log a numeric metric with formatting.

        Metrics are scalar floats rendered with a configurable format
        specification (default four decimal places). Implementations
        may additionally record the value in a machine-readable
        metrics store for later aggregation.

        Args:
            name: Metric identifier. Should be unique within a section.
            value: Numeric value.
            fmt: Python format specification applied to ``value`` for
                display only. The stored value is unaffected.

        """

    @abstractmethod
    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any:
        """Run an operation and log its elapsed time.

        Implementations are expected to measure wall-clock time around
        the call to ``operation()``, log the elapsed time under
        ``name``, and return whatever ``operation`` returns unchanged.

        Args:
            name: Human-readable operation name.
            operation: Zero-argument callable to execute.

        Returns:
            The return value of ``operation``. Implementations must
            propagate exceptions transparently; the timing measurement
            should not swallow errors.

        """
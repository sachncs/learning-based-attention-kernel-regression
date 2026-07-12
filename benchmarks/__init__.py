"""Benchmarking suite for the LAKER package.

This package provides reproducible performance benchmarks for the
Learning-based Attention Kernel Regression (LAKER) system.  It covers
the main computational kernels, preconditioner construction, PCG solves,
full model fitting, and low-rank approximation comparisons.

Submodules
----------
run
    End-to-end performance benchmarks for LAKER critical paths.
baseline
    Comparison of current performance against recorded baseline numbers.
executor
    Standardised timing, warmup, and statistical aggregation primitives
    used by the other benchmark modules.
approximations
    Accuracy and speed comparisons between exact, Nyström, and random
    Fourier feature kernel approximations.
reproducible
    Deterministic benchmarks with fixed seeds and a markdown report
    generator.
"""

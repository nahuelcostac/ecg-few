"""Compatibility entry point."""

from ecg_few.protocols import loocv as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)

"""Compatibility entry point."""

from ecg_few.evaluation import audit as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()

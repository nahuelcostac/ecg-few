"""Compatibility entry point."""

from ecg_few.reporting import sim_real as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()

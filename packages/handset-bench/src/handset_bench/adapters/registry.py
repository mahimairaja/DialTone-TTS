"""Runtime resolution of adapters by dotted path."""

from __future__ import annotations

import importlib

__all__ = ["resolve"]


def resolve(dotted: str) -> type:
    """Import `module:attribute` and return the attribute."""
    if ":" not in dotted:
        raise ValueError(
            f"adapter path {dotted!r} must be of the form 'module:attribute', "
            "for example 'handset_bench.adapters.piper:PiperAdapter'"
        )
    module_name, _, attribute = dotted.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"adapter path {dotted!r} has an empty module or attribute")

    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {attribute!r}"
        ) from exc

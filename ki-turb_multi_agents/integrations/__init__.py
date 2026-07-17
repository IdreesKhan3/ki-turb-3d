"""Lazy CFD backend registry.

Backends are imported only when requested. This prevents solver integrations
from importing the agent tool package while physics-only validation is loading.
"""
from __future__ import annotations

from importlib import import_module
from typing import Dict, List, Tuple, Type

from .base import BackendError, BackendNotConfigured, CFDBackend, LocalCommandBackend

_BACKEND_PATHS: Dict[str, Tuple[str, str]] = {
    "openlb": ("integrations.openlb_backend", "OpenLBBackend"),
    "palabos": ("integrations.palabos_backend", "PalabosBackend"),
    "ansys": ("integrations.ansys_backend", "AnsysBackend"),
}


def available_backends() -> List[str]:
    return sorted(_BACKEND_PATHS)


def _load_backend_class(name: str) -> Type[CFDBackend]:
    key = (name or "").strip().lower()
    target = _BACKEND_PATHS.get(key)
    if target is None:
        raise BackendError(f"unknown backend '{name}'. Available: {', '.join(available_backends())}")
    module_name, class_name = target
    return getattr(import_module(module_name), class_name)


def get_backend(name: str, **kwargs) -> CFDBackend:
    return _load_backend_class(name)(**kwargs)


def __getattr__(name: str):
    aliases = {
        "OpenLBBackend": "openlb",
        "PalabosBackend": "palabos",
        "AnsysBackend": "ansys",
    }
    if name in aliases:
        return _load_backend_class(aliases[name])
    raise AttributeError(name)


__all__ = [
    "BackendError", "BackendNotConfigured", "CFDBackend", "LocalCommandBackend",
    "OpenLBBackend", "PalabosBackend", "AnsysBackend", "available_backends", "get_backend",
]

"""Map (flow, backend) to a concrete case factory."""

from __future__ import annotations

from schemas import CFDCase

from .flows.hit import make_openlb_hit_case


def make_case(flow: str, backend: str, **kwargs) -> CFDCase:
    flow = (flow or "").lower()
    backend = (backend or "").lower()

    if flow == "hit" and backend == "openlb":
        return make_openlb_hit_case(**kwargs)

    raise ValueError(f"No case factory for flow={flow!r}, backend={backend!r}")


def has_factory(flow: str, backend: str) -> bool:
    return (flow or "").lower() == "hit" and (backend or "").lower() == "openlb"

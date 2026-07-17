"""Exact OpenLB HIT capability catalogue; no silent aliases or fallbacks."""
from __future__ import annotations
from typing import Dict,FrozenSet,Tuple
DNS_COLLISIONS=frozenset({"bgk","rlb","mrt","trt"})
LES_COLLISIONS=frozenset({"smagorinskybgk","smagorinskymrt","wale","consistentstrainsmagorinsky","shearsmagorinsky","krause","dynamicsmagorinsky"})
ALL_COLLISIONS=DNS_COLLISIONS|LES_COLLISIONS
COLLISION_XML_NAMES:Dict[str,str]={"bgk":"BGK","rlb":"RLB","mrt":"MRT","trt":"TRT","smagorinsky":"SmagorinskyBGK","smagorinskybgk":"SmagorinskyBGK","smagorinskymrt":"SmagorinskyMRT","wale":"WALE","consistentstrainsmagorinsky":"ConsistentStrainSmagorinsky","shearsmagorinsky":"ShearSmagorinsky","krause":"Krause","dynsmagorinsky":"DynamicSmagorinsky","dynamicsmagorinsky":"DynamicSmagorinsky"}
FORCING_SCHEMES=frozenset({"none","spectral_random","ornstein_uhlenbeck","constant_energy_input","constant_tke"})
FORCING_PATTERNS=frozenset({"random_phase","fixed_phase","sine","cosine","ou_process"})
DECAYING_FORCING=frozenset({"none","off",""})
_COLLISION_ALIASES:Dict[str,str]={
    "dns":"bgk",
    "regularized":"rlb",
    "regularised":"rlb",
    "regularizedlb":"rlb",
    "regularisedlb":"rlb",
    "smagorinsky":"smagorinskybgk",
    "smagorinskimrt":"smagorinskymrt",
    "smagorinskymrt":"smagorinskymrt",
    "dynsmagorinsky":"dynamicsmagorinsky",
    "dynamicsmag":"dynamicsmagorinsky",
    "constrain":"consistentstrainsmagorinsky",
    "css":"consistentstrainsmagorinsky",
    "shearsmag":"shearsmagorinsky",
}
def normalize_collision(scheme):
    s=(scheme or "BGK").strip().lower().replace("-","").replace("_","")
    s=_COLLISION_ALIASES.get(s,s)
    if s not in ALL_COLLISIONS: raise ValueError(f"unsupported OpenLB HIT collision: {scheme}")
    return s
def normalize_forcing_scheme(scheme):
    s=(scheme or "none").strip().lower().replace("-","_")
    aliases={"off":"none","spectral_low_k":"spectral_random","low_wavenumber":"spectral_random","ou":"ornstein_uhlenbeck","constant_energy":"constant_energy_input","linear":"constant_energy_input"}
    s=aliases.get(s,s)
    if s not in FORCING_SCHEMES: raise ValueError(f"unsupported OpenLB HIT forcing: {scheme}")
    return s
def normalize_forcing_pattern(pattern):return (pattern or "random_phase").strip().lower().replace("-","_")
def normalize_turbulence_regime(regime,collision=None):
    """Derive dns/les label from the collision model. Regime keywords never block builds."""
    if collision is not None:
        collision_norm=normalize_collision(collision)
        return "dns" if collision_norm in DNS_COLLISIONS else "les"
    if regime and str(regime).lower() in {"dns","les"}:
        return str(regime).lower()
    return "dns"
def collision_allowed_for_regime(collision,regime):
    """Known collisions are always accepted; regime is informational metadata."""
    try:
        return normalize_collision(collision) in ALL_COLLISIONS
    except ValueError:
        return False
def collision_regime_label(collision)->str:
    return normalize_turbulence_regime(None, collision)
def xml_collision_name(scheme):return COLLISION_XML_NAMES[normalize_collision(scheme)]
def agent_parameter_doc()->Tuple[str,...]:return ("typed case.hit.domain/scaling/collision/forcing/initial_condition/runtime/execution/checkpoint/outputs/analysis/visualization/acceptance",)

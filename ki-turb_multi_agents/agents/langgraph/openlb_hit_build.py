"""OpenLB HIT/FHIT build helpers: parameter catalog, parsing, and normalization."""
from __future__ import annotations

import re
from typing import Any

from schemas.cfd_case import normalize_hit_mode

_LOCKED_BUILD_FIELDS = frozenset({
    "reynolds_number",
    "viscosity",
    "relaxation_time",
    "mach_number",
    "char_velocity",
    "target_urms",
    "turbulence_regime",
    "scheme",
    "ic_wavenumber_min",
    "ic_wavenumber_max",
    "forcing_wavenumber_min",
    "forcing_wavenumber_max",
    "forcing_amplitude",
    "max_steps",
    "output_interval",
})

_LES_SCHEME_PATTERN = re.compile(
    r"\b("
    r"smagorinsky|wale|dynamicsmagorinsky|dynsmagorinsky|"
    r"consistentstrainsmagorinsky|shearsmagorinsky|krause|smagorinskimrt"
    r")\b",
    re.I,
)

_SCHEME_PATTERNS = (
    ("DynamicSmagorinsky", re.compile(r"\b(?:dynamic\s*smagorinsky|dynsmagorinsky)\b", re.I)),
    ("ConsistentStrainSmagorinsky", re.compile(
        r"\b(?:consistent\s*strain\s*smagorinsky|consistentstrainsmagorinsky|css\s*smagorinsky)\b",
        re.I,
    )),
    ("ShearSmagorinsky", re.compile(r"\b(?:shear\s*smagorinsky|shearsmagorinsky)\b", re.I)),
    ("Krause", re.compile(r"\bkrause\b", re.I)),
    ("Smagorinsky", re.compile(r"\bsmagorinsky\b", re.I)),
    ("WALE", re.compile(r"\bwale\b", re.I)),
    ("SmagorinskyMRT", re.compile(r"\bsmagorinsky\s*mrt\b", re.I)),
    ("MRT", re.compile(r"\bmrt\b", re.I)),
    ("TRT", re.compile(r"\btrt\b", re.I)),
    ("RLB", re.compile(r"\b(?:rlb|regulari[sz]ed(?:\s+lb)?)\b", re.I)),
    ("BGK", re.compile(r"\bbgk\b", re.I)),
)


def infer_locked_build_fields(args: dict[str, Any]) -> frozenset[str]:
    """Fields explicitly set by the caller must not be overwritten by calibration."""
    locked: set[str] = set()
    if args.get("reynolds_number") is not None:
        locked.add("reynolds_number")
    if args.get("viscosity") is not None:
        locked.add("viscosity")
    if args.get("relaxation_time") is not None:
        locked.add("relaxation_time")
    if args.get("mach_number") is not None:
        locked.add("mach_number")
    if args.get("char_velocity") is not None:
        locked.add("char_velocity")
    if args.get("turbulence_regime") is not None:
        locked.add("turbulence_regime")
    if args.get("scheme") is not None:
        locked.add("scheme")
    if args.get("ic_wavenumber_min") is not None:
        locked.add("ic_wavenumber_min")
    if args.get("ic_wavenumber_max") is not None:
        locked.add("ic_wavenumber_max")
    if args.get("forcing_wavenumber_min") is not None:
        locked.add("forcing_wavenumber_min")
    if args.get("forcing_wavenumber_max") is not None:
        locked.add("forcing_wavenumber_max")
    if args.get("forcing_amplitude") is not None:
        locked.add("forcing_amplitude")
    if args.get("target_urms") is not None:
        locked.add("target_urms")
    if args.get("max_steps") is not None:
        locked.add("max_steps")
    if args.get("output_interval") is not None:
        locked.add("output_interval")
    return frozenset(field for field in locked if field in _LOCKED_BUILD_FIELDS)


def normalize_build_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize aliases; collision model is authoritative, regime is derived."""
    from integrations.openlb_hit_catalog import normalize_collision, normalize_turbulence_regime

    result = {key: value for key, value in args.items() if value is not None}
    if "hit_mode" in result:
        result["hit_mode"] = normalize_hit_mode(result["hit_mode"])
    if result.get("forcing_scheme") and not result.get("forcing_type"):
        result["forcing_type"] = result.pop("forcing_scheme")
    elif "forcing_scheme" in result:
        result.pop("forcing_scheme", None)

    scheme = result.get("scheme")
    if scheme is not None:
        if str(scheme).strip().upper() == "DNS":
            result["scheme"] = "BGK"
            scheme = "BGK"
        normalize_collision(scheme)
        result["turbulence_regime"] = normalize_turbulence_regime(None, scheme)
    elif re.search(r"\bdns\b", str(result.get("turbulence_regime", "")).lower()):
        result.setdefault("scheme", "BGK")
        result["turbulence_regime"] = "dns"
    elif result.get("turbulence_regime"):
        result["turbulence_regime"] = str(result["turbulence_regime"]).lower()
    return result


def build_simulation_parameter_catalog() -> str:
    """Human-readable catalog of every build_simulation_case argument."""
    from agents.tools.simulation.case_builder import get_tool_definitions

    properties = get_tool_definitions()[0]["parameters"]["properties"]
    lines = [
        "Use build_simulation_case(backend, name, ...) with any of:",
    ]
    for name in sorted(properties):
        spec: dict[str, Any] = properties[name]
        desc = spec.get("description") or spec.get("type", "value")
        lines.append(f"  - {name}: {desc}")
    lines.extend([
        "",
        "Advanced: pass a full typed OpenLBHITConfig under `case` for lattice (D3Q19/D3Q27),",
        "domain size, periodic box, physical/lattice scaling, collision model, forcing,",
        "initial condition spectrum, runtime, outputs, execution mode (serial/MPI/OpenMP), etc.",
    ])
    return "\n".join(lines)


def simulation_build_step_instruction(user_request: str) -> str:
    """Instruction for the simulation agent to map user language → build_simulation_case args."""
    return (
        "Build exactly one OpenLB HIT or FHIT case with build_simulation_case.\n"
        "Translate the user request into tool arguments — do not ignore requested physics.\n"
        "Rules:\n"
        "- Call build_simulation_case exactly once, then stop.\n"
        "- Do NOT compile, start, fetch, or postprocess in this step.\n"
        "- Use only parameters the user asked for; do not invent unspecified Re, grid, tau, etc.\n"
        "- hit_mode must be `forced` (FHIT) or `decaying` (DHIT) — never `fhit` or `dhit`.\n"
        "- Grid: uniform N³, N^3, or explicit [nx, ny, nz].\n"
        "- Viscosity: viscosity / nu / kinematic viscosity. Relaxation: tau / relaxation time.\n"
        "- Reynolds: Re / Reynolds number. Lattice: D3Q19 or D3Q27 (via case or domain).\n"
        "- Collision: pass the exact `scheme` the user requests — BGK, DNS, RLB/regularized, "
        "MRT, TRT, Smagorinsky, WALE, ConsistentStrainSmagorinsky, ShearSmagorinsky, Krause, "
        "SmagorinskyMRT, DynamicSmagorinsky. Do not substitute a different model.\n"
        "- turbulence_regime (dns/les) is derived from the collision for metadata only; "
        "it never blocks a requested collision.\n"
        "- Output: max_steps and output_interval (e.g. save every 1000 iterations).\n"
        "- Units: honour physical-unit phrasing; otherwise use consistent lattice inputs.\n\n"
        f"Original user request:\n{user_request.strip()}"
    )


def _extract_resolution(text: str) -> list[int] | None:
    lower = text.lower()
    cube = re.search(r"(\d+)\s*(?:\^|×|x)\s*3\b", lower)
    if cube:
        n = int(cube.group(1))
        return [n, n, n]
    grid = re.search(r"(\d+)\s*grid\b", lower)
    if grid:
        n = int(grid.group(1))
        return [n, n, n]
    explicit = re.search(r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", lower)
    if explicit:
        return [int(explicit.group(1)), int(explicit.group(2)), int(explicit.group(3))]
    bracket = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", lower)
    if bracket:
        return [int(bracket.group(1)), int(bracket.group(2)), int(bracket.group(3))]
    return None


def _extract_scheme(text: str) -> str | None:
    schemes = extract_known_schemes(text)
    return schemes[0] if schemes else None


# Case A: MRT / collision=BGK — not prose like "except collision: N=16^3".
_LABELED_COLLISION = re.compile(
    r"(?i)(?:\bcase\s+[a-z0-9]+\s*:\s*([A-Za-z][A-Za-z0-9_-]*)"
    r"|\b(?:collision|scheme)\s*=\s*([A-Za-z][A-Za-z0-9_-]*))"
)


def extract_known_schemes(text: str) -> list[str]:
    """Known OpenLB HIT collisions mentioned in text, first-occurrence order."""
    hits: list[tuple[int, str]] = []
    for scheme, pattern in _SCHEME_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append((match.start(), scheme))
    hits.sort(key=lambda item: item[0])
    ordered: list[str] = []
    seen: set[str] = set()
    for _, scheme in hits:
        if scheme not in seen:
            ordered.append(scheme)
            seen.add(scheme)
    return ordered


def extract_collision_labels(text: str) -> list[str]:
    """Collision labels from Case/collision/scheme assignments + known scheme tokens."""
    hits: list[tuple[int, str]] = []
    for match in _LABELED_COLLISION.finditer(text or ""):
        label = match.group(1) or match.group(2)
        if label:
            hits.append((match.start(), label))
    for scheme, pattern in _SCHEME_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append((match.start(), scheme))
    hits.sort(key=lambda item: item[0])
    ordered: list[str] = []
    seen: set[str] = set()
    for _, label in hits:
        key = str(label).strip().lower()
        if key and key not in seen:
            ordered.append(str(label).strip())
            seen.add(key)
    return ordered


def _preferred_scheme_name(label: str) -> str | None:
    """Return build-facing scheme name if supported; else None."""
    from integrations.openlb_hit_catalog import normalize_collision, xml_collision_name

    raw = str(label or "").strip()
    if not raw:
        return None
    try:
        normalize_collision(raw)
    except ValueError:
        return None
    for scheme, pattern in _SCHEME_PATTERNS:
        if scheme.lower() == raw.lower() or pattern.fullmatch(raw):
            return scheme
    return xml_collision_name(raw)


def partition_collision_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    """Split labels into (supported build scheme names, unsupported raw labels)."""
    supported: list[str] = []
    unsupported: list[str] = []
    seen_ok: set[str] = set()
    seen_bad: set[str] = set()
    for label in labels:
        name = _preferred_scheme_name(label)
        if name is None:
            key = str(label).strip().lower()
            if key and key not in seen_bad:
                unsupported.append(str(label).strip())
                seen_bad.add(key)
            continue
        if name not in seen_ok:
            supported.append(name)
            seen_ok.add(name)
    return supported, unsupported


def _extract_forcing(text: str) -> tuple[str | None, int | None, int | None]:
    lower = text.lower()
    forcing_type = None
    if re.search(r"spectral(?:\s+low\s*k|\s+forcing)?", lower) or "spectral_low_k" in lower:
        forcing_type = "spectral_low_k"
    elif re.search(r"\blinear\s+forcing\b", lower):
        forcing_type = "linear"
    elif re.search(r"ornstein|ou\s+forcing", lower):
        forcing_type = "ornstein_uhlenbeck"

    band = re.search(r"k\s*[=:]\s*(\d+)\s*[-–]\s*(\d+)", lower)
    if band:
        return forcing_type or "spectral_low_k", int(band.group(1)), int(band.group(2))
    return forcing_type, None, None


def _auto_case_name(args: dict[str, Any]) -> str:
    parts: list[str] = []
    mode = args.get("hit_mode")
    if mode == "forced":
        parts.append("FHIT")
    elif mode == "decaying":
        parts.append("DHIT")
    else:
        parts.append("HIT")

    resolution = args.get("resolution")
    if isinstance(resolution, (list, tuple)) and resolution:
        if len(resolution) == 3 and resolution[0] == resolution[1] == resolution[2]:
            parts.append(str(resolution[0]))
        else:
            parts.append("x".join(str(n) for n in resolution))

    scheme = args.get("scheme")
    if scheme:
        parts.append(str(scheme))

    forcing = args.get("forcing_type")
    if forcing:
        parts.append(str(forcing).replace("_", ""))

    return "_".join(parts) if parts else "openlb_hit"


def _extract_output_interval(text: str) -> int | None:
    lower = text.lower()
    patterns = (
        r"(?:every|each)\s+(\d+)\s*(?:iterations?|time\s*steps?|lattice\s*steps?|steps?)",
        r"(?:save|write|output|dump).*(?:every|each)\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return int(match.group(1))
    return None


def _extract_max_steps(text: str) -> int | None:
    lower = text.lower()
    output_spans = [
        match.span()
        for pattern in (
            r"(?:every|each)\s+\d+\s*(?:iterations?|time\s*steps?|lattice\s*steps?|steps?)",
            r"(?:save|write|output|dump).*(?:every|each)\s+\d+",
        )
        for match in re.finditer(pattern, lower)
    ]

    def _inside_output_clause(start: int, end: int) -> bool:
        return any(span_start <= start and end <= span_end for span_start, span_end in output_spans)

    candidates: list[tuple[int, int, int]] = []

    for match in re.finditer(
        r"(?:for\s+)?(\d+)\s*(?:iterations?|time\s*steps?|lattice\s*steps?|steps)\b",
        lower,
    ):
        if not _inside_output_clause(match.start(), match.end()):
            candidates.append((match.start(), int(match.group(1)), 0))

    for match in re.finditer(
        r"(?:iterations?|time\s*steps?|lattice\s*steps?|steps)\s*[=:]?\s*(\d+)\b",
        lower,
    ):
        if not _inside_output_clause(match.start(), match.end()):
            candidates.append((match.start(), int(match.group(1)), 1))

    if not candidates:
        return None
    # Prefer explicit totals like "iterations 1000" over incidental smaller counts.
    candidates.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return candidates[0][1]


def has_explicit_openlb_case_params(text: str) -> bool:
    """True when the user text itself specifies CFD case parameters (no calibrated defaults)."""
    text = text or ""
    lower = text.lower()
    if _extract_resolution(text) is not None:
        return True
    if _extract_scheme(text) is not None or _LES_SCHEME_PATTERN.search(text):
        return True
    if _extract_max_steps(text) is not None or _extract_output_interval(text) is not None:
        return True
    forcing_type, k_min, k_max = _extract_forcing(text)
    if forcing_type or k_min is not None or k_max is not None:
        return True
    if re.search(r"(?:re|reynolds(?:\s+number)?)\s*(?:[=:]\s*|\s+)\d", lower):
        return True
    if re.search(r"(?:nu|viscosity|τ|tau|relaxation(?:\s+time)?|mach)\s*[=:]?\s*\d", lower):
        return True
    if re.search(r"\b(?:dns|les)\b", lower):
        return True
    return False


def parse_openlb_build_args(text: str) -> dict[str, Any]:
    """Extract build_simulation_case kwargs from natural-language HIT/FHIT requests."""
    text = (text or "").strip()
    lower = text.lower()
    locked: set[str] = set()

    args: dict[str, Any] = {
        "backend": "openlb",
        "flow": "hit",
    }

    if re.search(r"\bdhit\b", lower):
        args["hit_mode"] = "decaying"
    elif re.search(r"\bfhit\b", lower) or re.search(r"\bhit\b", lower):
        args["hit_mode"] = "forced"
    elif re.search(r"\bopenlb\b", lower) and re.search(
        r"\b(simulate|simulation|smoke|compile|run|fetch|solver|set|configure|create|grid|iterations?)\b",
        lower,
    ):
        args["hit_mode"] = "forced"

    resolution = _extract_resolution(text)
    if resolution:
        args["resolution"] = resolution

    re_match = re.search(
        r"(?:re|reynolds(?:\s+number)?)\s*(?:[=:]\s*|\s+)(\d+(?:\.\d+)?)",
        lower,
    )
    if re_match:
        args["reynolds_number"] = float(re_match.group(1))
        locked.add("reynolds_number")

    nu_match = re.search(
        r"(?:nu|viscosity|ν)\s*[=:]\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        lower,
    )
    if nu_match:
        args["viscosity"] = float(nu_match.group(1))
        locked.add("viscosity")

    tau_match = re.search(
        r"(?:tau|relaxation(?:\s+time)?)\s*(?:[=:]\s*|\s+)(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        lower,
    )
    if tau_match:
        args["relaxation_time"] = float(tau_match.group(1))
        locked.add("relaxation_time")

    mach_match = re.search(r"(?:ma|mach)(?:\s+number)?\s*[=:]\s*(\d+(?:\.\d+)?)", lower)
    if mach_match:
        args["mach_number"] = float(mach_match.group(1))
        locked.add("mach_number")

    scheme = _extract_scheme(text)

    if re.search(r"\bdns\b", lower):
        args["turbulence_regime"] = "dns"
        locked.add("turbulence_regime")
        if scheme is None:
            args["scheme"] = "BGK"
            locked.add("scheme")
        else:
            args["scheme"] = scheme
            locked.add("scheme")
    elif re.search(r"\bles\b", lower):
        args["turbulence_regime"] = "les"
        locked.add("turbulence_regime")
        if scheme:
            args["scheme"] = scheme
            locked.add("scheme")
    elif scheme:
        args["scheme"] = scheme
        locked.add("scheme")
    elif _LES_SCHEME_PATTERN.search(text):
        args["turbulence_regime"] = "les"

    forcing_type, k_min, k_max = _extract_forcing(text)
    if forcing_type:
        args["forcing_type"] = forcing_type
    if k_min is not None:
        args["forcing_wavenumber_min"] = k_min
        locked.add("forcing_wavenumber_min")
    if k_max is not None:
        args["forcing_wavenumber_max"] = k_max
        locked.add("forcing_wavenumber_max")

    output_interval = _extract_output_interval(text)
    if output_interval is not None:
        args["output_interval"] = output_interval
        locked.add("output_interval")

    max_steps = _extract_max_steps(text)
    if max_steps is not None:
        args["max_steps"] = max_steps
        locked.add("max_steps")

    args = normalize_build_args(args)

    args["name"] = _auto_case_name(args)

    from agents.physics_constraint_agent import PhysicsConstraintAgent, _OPENLB_HIT_CONFIG_KEY

    calibrated = PhysicsConstraintAgent().calibrate_build_args(
        args,
        locked=frozenset(field for field in locked if field in _LOCKED_BUILD_FIELDS),
    )
    calibrated.pop(_OPENLB_HIT_CONFIG_KEY, None)
    return normalize_build_args(calibrated)


__all__ = [
    "build_simulation_parameter_catalog",
    "simulation_build_step_instruction",
    "normalize_hit_mode",
    "normalize_build_args",
    "infer_locked_build_fields",
    "has_explicit_openlb_case_params",
    "parse_openlb_build_args",
    "extract_known_schemes",
    "extract_collision_labels",
    "partition_collision_labels",
]

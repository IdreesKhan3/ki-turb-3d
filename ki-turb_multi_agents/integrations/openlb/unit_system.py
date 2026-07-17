"""Fill the solver-neutral UnitSystem from an OpenLB HIT configuration."""
from __future__ import annotations

from schemas.openlb_hit import OpenLBHITConfig
from schemas.unit_system import FieldUnit, UnitFrame, UnitSystem


def unit_system_from_openlb_hit(config: OpenLBHITConfig) -> UnitSystem:
    """Map OpenLB HIT scaling + known output conventions into UnitSystem.

    OpenLB kiTurbHIT3D writes velocity VTI via SuperLatticePhysVelocity3D (physical),
    density as lattice ρ, forcing default as lattice acceleration, time as physical.
    """
    derived = config.derive_scaling()
    forcing_units = getattr(config.forcing, "units", "lattice_acceleration")
    forcing_frame = (
        UnitFrame.PHYSICAL
        if str(forcing_units) == "physical_acceleration"
        else UnitFrame.LATTICE
    )
    fields = {
        "velocity_field": FieldUnit(
            kind="velocity_field",
            frame=UnitFrame.PHYSICAL,
            label="physical_velocity",
            notes="SuperLatticePhysVelocity3D",
        ),
        "vorticity_field": FieldUnit(
            kind="vorticity_field",
            frame=UnitFrame.PHYSICAL,
            label="1/physical_time",
            notes="from physical velocity and dx",
        ),
        "density_field": FieldUnit(
            kind="density_field",
            frame=UnitFrame.LATTICE,
            label="lattice_density",
            notes="raw lattice ρ (not converted)",
        ),
        "pressure_field": FieldUnit(
            kind="pressure_field",
            frame=UnitFrame.LATTICE,
            label="lattice_pressure",
            notes="(ρ-ρ0)/cs² style lattice pressure when written",
        ),
        "forcing_field": FieldUnit(
            kind="forcing_field",
            frame=forcing_frame,
            label=str(forcing_units),
        ),
        "time": FieldUnit(
            kind="time",
            frame=UnitFrame.PHYSICAL,
            label="physical_time",
            notes="diagnostics physical_time / UnitConverter.getPhysTime",
        ),
        "length": FieldUnit(kind="length", frame=UnitFrame.PHYSICAL, label="physical_length"),
        "viscosity": FieldUnit(
            kind="viscosity",
            frame=UnitFrame.PHYSICAL,
            label="physical_viscosity",
        ),
    }
    return UnitSystem(
        source_backend="openlb",
        analysis_frame=UnitFrame.PHYSICAL,
        length_ref=derived.characteristic_length,
        velocity_ref=derived.characteristic_velocity,
        density_ref=float(config.scaling.density),
        viscosity=derived.physical_viscosity,
        reynolds_number=derived.reynolds_number,
        dx=derived.dx,
        dt=derived.dt,
        lattice_sound_speed=float(config.scaling.lattice_sound_speed),
        relaxation_time=derived.relaxation_time,
        mach=derived.actual_mach,
        fields=fields,
        metadata={
            "openlb_converter": "UnitConverterFromResolutionAndRelaxationTime",
            "forcing_units": str(forcing_units),
            "lattice": config.domain.lattice.value
            if hasattr(config.domain.lattice, "value")
            else str(config.domain.lattice),
        },
    )


__all__ = ["unit_system_from_openlb_hit"]

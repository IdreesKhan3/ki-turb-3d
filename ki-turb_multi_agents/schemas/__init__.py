"""Structured schemas for solver-neutral CFD workflows."""
from .cfd_case import *
from .simulation_job import SimulationJob, JobStatus, JobPaths
from .dataset_manifest import DatasetManifest, DatasetFile, STANDARD_FILE_KINDS
from .validation_report import ValidationReport, ConstraintCheck
from .openlb_hit import *
from .hit_analysis_products import *
from .unit_system import UnitFrame, FieldKind, FieldUnit, UnitSystem

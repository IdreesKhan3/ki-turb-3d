"""Turn raw solver velocity fields into KI-TURB-ready turbulence quantities.

The pipeline reads velocity snapshots from a dataset manifest, computes spectra,
isotropy, real-space statistics, flatness, structure functions, and PDFs, and
writes them under ``<base_dir>/processed`` using the filenames KI-TURB detects.
"""

from .pipeline import postprocess_manifest

__all__ = ["postprocess_manifest"]

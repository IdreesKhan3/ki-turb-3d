"""
Data readers module
Contains all file parsing functions
"""

from .csv_reader import read_csv_data, read_eps_validation_csv
from .parameter_reader import read_parameters, format_parameters_for_display
from .spectrum_reader import read_spectrum_file
from .binary_reader import read_structure_function_file, read_tau_analysis_file
from .text_reader import read_structure_function_txt, read_flatness_file
from .norm_spectrum_reader import read_norm_spectrum_file
from .vti_reader import read_vti_file, compute_velocity_magnitude, compute_vorticity
from .hdf5_reader import read_hdf5_file, read_hdf5_file_fortran_order

__all__ = [
    'read_csv_data',
    'read_eps_validation_csv',
    'read_parameters',
    'format_parameters_for_display',
    'read_spectrum_file',
    'read_structure_function_file',
    'read_tau_analysis_file',
    'read_structure_function_txt',
    'read_flatness_file',
    'read_norm_spectrum_file',
    'read_vti_file',
    'read_hdf5_file',
    'read_hdf5_file_fortran_order',
    'compute_velocity_magnitude',
    'compute_vorticity',
]


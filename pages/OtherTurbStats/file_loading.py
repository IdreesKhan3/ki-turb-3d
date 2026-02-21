"""
Other Turbulence Stats — File discovery, data loading, session state.
"""

import glob
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from data_readers.csv_reader import read_csv_data
from utils.file_detector import detect_simulation_files
from utils.plot_style import default_plot_style


def init_session_state():
    """Initialize session state for Other Turbulence Stats page."""
    if "plot_style" not in st.session_state:
        st.session_state.plot_style = default_plot_style()
    if "custom_plot_traces" not in st.session_state:
        st.session_state.custom_plot_traces = []
    if "custom_plot_legend_names" not in st.session_state:
        st.session_state.custom_plot_legend_names = {}
    if "custom_plot_axis_labels" not in st.session_state:
        st.session_state.custom_plot_axis_labels = {"x": "X", "y": "Y"}


def _load_turbulence_stats(
    data_dirs: List[str],
    files: Dict[str, List],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]], Dict[str, Dict]]:
    """Load turbulence_stats CSV files. Returns (all_dataframes, available_columns, table_data)."""
    all_dataframes = {}
    available_columns = {}
    table_data = {}
    csv_files = files.get("real_turb_stats", [])

    if not csv_files:
        return all_dataframes, available_columns, table_data

    if len(data_dirs) > 1:
        for csv_file in csv_files:
            csv_path = Path(csv_file).resolve()
            dir_name = None
            for data_dir_path in data_dirs:
                data_dir_obj = Path(data_dir_path).resolve()
                try:
                    csv_path.relative_to(data_dir_obj)
                    dir_name = Path(data_dir_path).name
                    break
                except ValueError:
                    continue
            if not dir_name:
                dir_name = csv_path.parent.name
            try:
                df_stats = read_csv_data(str(csv_file))
                key = f"turbulence_stats_{dir_name}"
                all_dataframes[key] = df_stats
                available_columns[key] = list(df_stats.columns)
                table_data[key] = {"df": df_stats, "dir_name": dir_name, "type": "turbulence_stats"}
            except Exception as e:
                st.warning(f"Could not load {csv_path.name}: {e}")
    else:
        if len(csv_files) > 1:
            for idx, csv_file in enumerate(csv_files, 1):
                csv_path = Path(csv_file)
                file_suffix = csv_path.stem.replace("turbulence_stats", "").strip("_")
                key_suffix = f"_{file_suffix}" if file_suffix else f"_{idx}"
                try:
                    df_stats = read_csv_data(str(csv_file))
                    key = f"turbulence_stats{key_suffix}"
                    all_dataframes[key] = df_stats
                    available_columns[key] = list(df_stats.columns)
                    table_data[key] = {"df": df_stats, "dir_name": csv_path.stem, "type": "turbulence_stats"}
                except Exception as e:
                    st.warning(f"Could not load {csv_path.name}: {e}")
        else:
            try:
                df_stats = read_csv_data(str(csv_files[0]))
                all_dataframes["turbulence_stats"] = df_stats
                available_columns["turbulence_stats"] = list(df_stats.columns)
                table_data["turbulence_stats"] = {"df": df_stats, "dir_name": None, "type": "turbulence_stats"}
            except Exception as e:
                st.warning(f"Could not load turbulence stats: {e}")
    return all_dataframes, available_columns, table_data


def _load_eps_validation(
    data_dirs: List[str],
    data_dir: Path,
    files: Dict[str, List],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]], Dict[str, Dict]]:
    """Load eps_real_validation / turbulence_validation CSV files."""
    all_dataframes = {}
    available_columns = {}
    table_data = {}
    eps_files = files.get("spectral_turb_stats", [])
    if not eps_files:
        eps_files = glob.glob(str(data_dir / "eps_real_validation*.csv")) + glob.glob(
            str(data_dir / "turbulence_validation*.csv")
        )

    if not eps_files:
        return all_dataframes, available_columns, table_data

    if len(data_dirs) > 1:
        for eps_file in eps_files:
            eps_path = Path(eps_file).resolve()
            dir_name = None
            for data_dir_path in data_dirs:
                data_dir_obj = Path(data_dir_path).resolve()
                try:
                    eps_path.relative_to(data_dir_obj)
                    dir_name = Path(data_dir_path).name
                    break
                except ValueError:
                    continue
            if not dir_name:
                dir_name = eps_path.parent.name
            try:
                df_val = pd.read_csv(str(eps_file))
                key = f"eps_validation_{dir_name}"
                all_dataframes[key] = df_val
                available_columns[key] = list(df_val.columns)
                table_data[key] = {"df": df_val, "dir_name": dir_name, "type": "eps_validation"}
            except Exception as e:
                st.warning(f"Could not load {eps_path.name} from {dir_name}: {e}")
    else:
        if len(eps_files) > 1:
            for idx, eps_file in enumerate(eps_files, 1):
                eps_path = Path(eps_file)
                file_suffix = eps_path.stem.replace("eps_real_validation", "").strip("_")
                key_suffix = f"_{file_suffix}" if file_suffix else f"_{idx}"
                try:
                    df_val = pd.read_csv(str(eps_file))
                    key = f"eps_validation{key_suffix}"
                    all_dataframes[key] = df_val
                    available_columns[key] = list(df_val.columns)
                    table_data[key] = {"df": df_val, "dir_name": eps_path.stem, "type": "eps_validation"}
                except Exception as e:
                    st.warning(f"Could not load {eps_path.name}: {e}")
        else:
            try:
                df_val = pd.read_csv(str(eps_files[0]))
                all_dataframes["eps_validation"] = df_val
                available_columns["eps_validation"] = list(df_val.columns)
                table_data["eps_validation"] = {"df": df_val, "dir_name": None, "type": "eps_validation"}
            except Exception:
                pass
    return all_dataframes, available_columns, table_data


def load_all_data() -> Optional[Tuple[Path, List[str], Dict[str, pd.DataFrame], Dict[str, Dict]]]:
    """
    Load turbulence stats and eps validation from all data directories.
    Returns (data_dir, data_dirs, all_dataframes, table_data) or None on early exit.
    """
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]

    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return None

    data_dir = Path(data_dirs[0])
    all_files_dict = {}
    for data_dir_path in data_dirs:
        data_dir_obj = Path(data_dir_path)
        if data_dir_obj.exists():
            dir_files = detect_simulation_files(str(data_dir_obj))
            for file_type, file_list in dir_files.items():
                if file_type not in all_files_dict:
                    all_files_dict[file_type] = []
                all_files_dict[file_type].extend(
                    [str(f) if isinstance(f, Path) else f for f in file_list]
                )

    all_dataframes = {}
    available_columns = {}
    table_data = {}

    df_turb, col_turb, tbl_turb = _load_turbulence_stats(data_dirs, all_files_dict)
    all_dataframes.update(df_turb)
    available_columns.update(col_turb)
    table_data.update(tbl_turb)

    df_eps, col_eps, tbl_eps = _load_eps_validation(data_dirs, data_dir, all_files_dict)
    all_dataframes.update(df_eps)
    available_columns.update(col_eps)
    table_data.update(tbl_eps)

    if not all_dataframes:
        st.info("No CSV files found. Please load data from the Overview page.")
        return None

    return (data_dir, data_dirs, all_dataframes, table_data)

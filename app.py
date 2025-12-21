"""
Turbulence Statistics Dashboard
Main Streamlit application entry point
"""

import streamlit as st
from pathlib import Path
import sys
import glob
import logging

# Add project root to path (for direct execution)
# If installed as package, this is not needed
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.theme_config import get_theme_list, inject_theme_css, get_theme
from utils.file_detector import detect_simulation_files, natural_sort_key

# App version for logging
APP_VERSION = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Log app startup with version info
def _get_version_info():
    """Get version info including git commit if available."""
    version_info = f"KI-TURB 3D v{APP_VERSION}"
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            version_info += f" (commit: {commit_hash})"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass  # Git not available or not a git repo
    return version_info

logger.info(f"Starting {_get_version_info()}")
logger.info(f"Project root: {project_root}")

# Page configuration
st.set_page_config(
    page_title="KI-TURB 3D",
    page_icon="⚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_directory' not in st.session_state:
    st.session_state.data_directory = None
if 'data_directories' not in st.session_state:
    st.session_state.data_directories = []  # List of multiple directories
if 'multi_directory_mode' not in st.session_state:
    st.session_state.multi_directory_mode = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'theme' not in st.session_state:
    st.session_state.theme = "Light Scientific"
if 'all_loaded_files' not in st.session_state:
    st.session_state.all_loaded_files = {}


# ==========================================================
# Helper Functions
# ==========================================================

def _get_velocity_files(data_dir: Path):
    """Scan directory for velocity files (.vti, .h5, .hdf5) with case-insensitive matching."""
    vti_files = sorted(
        glob.glob(str(data_dir / "*.vti")) + glob.glob(str(data_dir / "*.VTI")),
        key=natural_sort_key
    )
    hdf5_files = sorted(
        glob.glob(str(data_dir / "*.h5")) + glob.glob(str(data_dir / "*.H5")) +
        glob.glob(str(data_dir / "*.hdf5")) + glob.glob(str(data_dir / "*.HDF5")),
        key=natural_sort_key
    )
    return vti_files + hdf5_files


def _scan_directory_for_files(data_dir_path: str):
    """
    Scan a directory and collect all simulation files.
    
    Returns:
        Dictionary mapping file types to lists of file info dicts with keys:
        'full_path', 'directory', 'filename'
    """
    data_dir = Path(data_dir_path)
    all_files_by_type = {}
    
    # Use file_detector for standard simulation files
    files_dict = detect_simulation_files(str(data_dir))
    for file_type, file_list in files_dict.items():
        if file_type not in all_files_by_type:
            all_files_by_type[file_type] = []
        for file_path in file_list:
            all_files_by_type[file_type].append({
                'full_path': str(Path(file_path)),
                'directory': str(data_dir),
                'filename': Path(file_path).name
            })
    
    # Collect velocity files separately
    velocity_files = _get_velocity_files(data_dir)
    if 'velocity_files' not in all_files_by_type:
        all_files_by_type['velocity_files'] = []
    for f in velocity_files:
        all_files_by_type['velocity_files'].append({
            'full_path': str(f),
            'directory': str(data_dir),
            'filename': Path(f).name
        })
    
    return all_files_by_type


def _load_directories_and_scan(valid_dirs: list):
    """Load multiple directories and scan all files, updating session state."""
    if not valid_dirs:
        logger.warning("_load_directories_and_scan called with empty directory list")
        return False
    
    logger.info(f"Loading {len(valid_dirs)} directory(ies): {[str(Path(d).name) for d in valid_dirs]}")
    st.session_state.data_directories = valid_dirs
    st.session_state.data_directory = valid_dirs[0]
    st.session_state.data_loaded = True
    
    # Scan all directories and collect all files
    all_files_by_type = {}
    for data_dir_path in valid_dirs:
        logger.info(f"Scanning directory: {data_dir_path}")
        dir_files = _scan_directory_for_files(data_dir_path)
        # Merge files from all directories
        for file_type, file_list in dir_files.items():
            if file_type not in all_files_by_type:
                all_files_by_type[file_type] = []
            all_files_by_type[file_type].extend(file_list)
        # Log file counts per type for this directory
        file_counts = {ft: len(fl) for ft, fl in dir_files.items() if fl}
        if file_counts:
            logger.info(f"Found files in {Path(data_dir_path).name}: {file_counts}")
    
    # Deduplicate by full_path
    for ft, lst in all_files_by_type.items():
        seen = set()
        dedup = []
        for item in lst:
            if item["full_path"] not in seen:
                dedup.append(item)
                seen.add(item["full_path"])
        all_files_by_type[ft] = dedup
    
    # Log total file counts after deduplication
    total_counts = {ft: len(fl) for ft, fl in all_files_by_type.items() if fl}
    logger.info(f"Total files loaded (after deduplication): {total_counts}")
    
    st.session_state.all_loaded_files = all_files_by_type
    return True


def _load_single_directory_and_merge(dir_path: str):
    """
    Load a single directory and merge its files into existing all_loaded_files.
    If in multi-directory mode, adds to data_directories. Otherwise replaces.
    """
    logger.info(f"Loading directory: {dir_path}")
    resolved = _resolve_directory_path(dir_path)
    if not resolved:
        logger.warning(f"Directory not found or invalid: {dir_path}")
        return False
    
    abs_path_str = str(resolved)
    logger.info(f"Resolved directory path: {abs_path_str}")
    st.session_state.data_directory = abs_path_str
    
    # Handle multi-directory mode
    if st.session_state.multi_directory_mode:
        # Add to existing directories if not already there
        if abs_path_str not in st.session_state.data_directories:
            st.session_state.data_directories.append(abs_path_str)
            logger.info(f"Added to multi-directory list. Total directories: {len(st.session_state.data_directories)}")
        # Merge files into existing all_loaded_files
        new_files = _scan_directory_for_files(abs_path_str)
        file_counts = {ft: len(fl) for ft, fl in new_files.items() if fl}
        if file_counts:
            logger.info(f"Scanned files: {file_counts}")
        if 'all_loaded_files' not in st.session_state or not st.session_state.all_loaded_files:
            st.session_state.all_loaded_files = new_files
        else:
            # Merge new files into existing
            for file_type, file_list in new_files.items():
                if file_type not in st.session_state.all_loaded_files:
                    st.session_state.all_loaded_files[file_type] = []
                st.session_state.all_loaded_files[file_type].extend(file_list)
        
        # Deduplicate by full_path after merging
        for ft, lst in st.session_state.all_loaded_files.items():
            seen = set()
            dedup = []
            for item in lst:
                if item["full_path"] not in seen:
                    dedup.append(item)
                    seen.add(item["full_path"])
            st.session_state.all_loaded_files[ft] = dedup
    else:
        # Single directory mode - replace
        st.session_state.data_directories = [abs_path_str]
        st.session_state.all_loaded_files = _scan_directory_for_files(abs_path_str)
        file_counts = {ft: len(fl) for ft, fl in st.session_state.all_loaded_files.items() if fl}
        if file_counts:
            logger.info(f"Loaded files: {file_counts}")
    
    st.session_state.data_loaded = True
    return True


def _resolve_directory_path(user_dir: str, log_warnings: bool = True):
    """
    Try to resolve a directory path (absolute or relative to project root).
    
    Args:
        user_dir: Directory path to resolve
        log_warnings: Whether to log warnings if directory not found
    
    Returns:
        Path object if found, None otherwise
    """
    if not user_dir:
        if log_warnings:
            logger.debug("Empty directory path provided")
        return None
    
    # Normalize path separators for cross-platform compatibility
    user_dir = user_dir.strip().replace('\\', '/')
    
    # Try as absolute path first
    abs_path = Path(user_dir).resolve()
    if abs_path.exists() and abs_path.is_dir():
        logger.debug(f"Resolved as absolute path: {abs_path}")
        return abs_path
    
    # Try as relative path from project root
    relative_path = (project_root / user_dir).resolve()
    if relative_path.exists() and relative_path.is_dir():
        logger.debug(f"Resolved as relative path: {relative_path}")
        return relative_path
    
    if log_warnings:
        logger.warning(f"Could not resolve directory path: {user_dir}")
        logger.debug(f"Project root: {project_root}")
        logger.debug(f"Tried absolute: {abs_path}")
        logger.debug(f"Tried relative: {relative_path}")
    return None


def _display_logo_or_title():
    """Display logo if available, otherwise show text title."""
    logo_path = project_root / "logo.png"
    if logo_path.exists() and logo_path.stat().st_size > 0:
        try:
            st.image(str(logo_path), width='stretch')
        except Exception:
            st.markdown("### KI-TURB 3D")
            st.caption("Turbulence Visualization & Analysis Suite")
    else:
        st.markdown("### KI-TURB 3D")
        st.caption("Turbulence Visualization & Analysis Suite")


def _get_theme_colors():
    """Get color scheme based on current theme."""
    current_theme = st.session_state.get("theme", "Light Scientific")
    theme_info = get_theme(current_theme)
    is_dark = "Dark" in current_theme
    
    if is_dark:
        return {
            'bg_color': theme_info['paper_bgcolor'],
            'card_bg': "#2d2d30",
            'text_color': theme_info['font_color'],
            'secondary_text': "#b0b0b0",
            'border_color': theme_info['grid_color'],
            'accent_color': "#4ec9b0"
        }
    else:
        return {
            'bg_color': "#f8f9fa",
            'card_bg': "#f8f9fa",
            'text_color': "#2c3e50",
            'secondary_text': "#5a6c7d",
            'border_color': "#e9ecef",
            'accent_color': "#2c3e50"
        }


def main():
    """Main application entry point"""
    
    # Apply theme CSS globally (must be first to apply to all pages)
    inject_theme_css()
    
    # Sidebar - Data Selection
    with st.sidebar:
        # Logo (if exists)
        _display_logo_or_title()
        st.markdown("---")
        
        # Theme Selector
        st.subheader("Theme")
        available_themes = get_theme_list()
        default_idx = available_themes.index(st.session_state.theme) if st.session_state.theme in available_themes else 0
        
        selected_theme = st.selectbox(
            "Select Theme",
            options=available_themes,
            index=default_idx,
            help="Choose a visualization theme optimized for different use cases",
            key="theme_selector"
        )
        
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            # Clear plot_style to force reapplication of theme
            if 'plot_style' in st.session_state:
                del st.session_state.plot_style
            st.rerun()
        
        # Show theme description
        theme_info = get_theme(selected_theme)
        st.caption(f"{theme_info['description']}")
        
        st.markdown("---")
        
        # Privacy Settings - Streamlit Telemetry Toggle
        with st.expander("⚙️ Privacy Settings", expanded=False):
            st.caption("**Streamlit Anonymous Telemetry**")
            
            # Check current telemetry setting from Streamlit config
            import os
            from pathlib import Path
            
            config_dir = Path.home() / ".streamlit"
            config_file = config_dir / "config.toml"
            
            # Read current setting from config file
            current_telemetry = True  # Default Streamlit behavior
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        if 'gatherUsageStats' in content:
                            # Parse the setting
                            import re
                            match = re.search(r'gatherUsageStats\s*=\s*(true|false)', content, re.IGNORECASE)
                            if match:
                                current_telemetry = match.group(1).lower() == 'true'
                except:
                    pass
            
            telemetry_enabled = st.checkbox(
                "Enable anonymous usage statistics",
                value=current_telemetry,
                help="Streamlit collects anonymous telemetry (version, OS, Python version). Your data and code are never sent. Disable for complete privacy.",
                key="telemetry_toggle"
            )
            
            # Update config file if changed
            if telemetry_enabled != current_telemetry:
                try:
                    # Create .streamlit directory if it doesn't exist
                    config_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Read existing config or create new
                    existing_config = ""
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            existing_config = f.read()
                    
                    # Update or add gatherUsageStats setting
                    import re
                    new_value = "true" if telemetry_enabled else "false"
                    
                    if 'gatherUsageStats' in existing_config:
                        # Replace existing setting
                        updated_config = re.sub(
                            r'gatherUsageStats\s*=\s*(true|false)',
                            f'gatherUsageStats = {new_value}',
                            existing_config,
                            flags=re.IGNORECASE
                        )
                    else:
                        # Add new setting
                        if '[browser]' in existing_config:
                            # Add under existing [browser] section
                            updated_config = re.sub(
                                r'\[browser\]',
                                f'[browser]\ngatherUsageStats = {new_value}',
                                existing_config,
                                count=1
                            )
                        else:
                            # Add new [browser] section
                            updated_config = existing_config + f"\n\n[browser]\ngatherUsageStats = {new_value}\n"
                    
                    # Write updated config
                    with open(config_file, 'w') as f:
                        f.write(updated_config)
                    
                    st.success(f"✅ Telemetry {'enabled' if telemetry_enabled else 'disabled'}. Please **restart the app** for changes to take effect.")
                    st.info("💡 Run: `streamlit run app.py` again")
                except Exception as e:
                    st.error(f"Failed to update config: {str(e)}")
            
            st.caption("ℹ️ This only affects Streamlit telemetry. Your turbulence data always stays local.")
        
        st.markdown("---")
        
        st.subheader("Data Selection")
        
        # Mode selection: Single or Multiple directories
        # Preserve current mode selection
        current_mode_index = 1 if st.session_state.multi_directory_mode else 0
        selection_mode = st.radio(
            "Selection Mode:",
            ["Single Simulation", "Multiple Simulations (Comparison)"],
            index=current_mode_index,
            help="Choose single directory for standard analysis, or multiple directories to compare simulations"
        )
        st.session_state.multi_directory_mode = (selection_mode == "Multiple Simulations (Comparison)")
        
        if st.session_state.multi_directory_mode:
            # Multiple directory mode
            st.markdown("**Select multiple simulation directories to compare:**")
            
            # Text area for multiple paths
            dirs_input = st.text_area(
                "Directory paths (one per line):",
                value="\n".join(st.session_state.data_directories) if st.session_state.data_directories else "",
                height=150,
                help="Enter absolute paths or paths relative to project root, one per line.\nExample:\nexamples/DNS/512\nexamples/DNS/256\nexamples/LES/128"
            )
            
            if st.button("Load Multiple Directories", type="primary"):
                dir_paths = [p.strip() for p in dirs_input.strip().split("\n") if p.strip()]
                valid_dirs = []
                
                for dir_path in dir_paths:
                    resolved = _resolve_directory_path(dir_path)
                    if resolved:
                        valid_dirs.append(str(resolved))
                    else:
                        st.warning(f"Directory not found: {dir_path}")
                
                if _load_directories_and_scan(valid_dirs):
                    st.success(f"✅ Loaded {len(valid_dirs)} directories")
                    for i, d in enumerate(valid_dirs, 1):
                        try:
                            rel_path = Path(d).relative_to(project_root)
                            st.caption(f"  {i}. {rel_path}")
                        except ValueError:
                            st.caption(f"  {i}. {d}")
                    st.rerun()
                else:
                    st.error("No valid directories found. Please check your paths.")
            
            # Quick multi-select from common locations
            st.markdown("---")
            st.markdown("**Quick Multi-Select:**")
            quick_dirs = [
                ("DNS/512", "examples/DNS/512"),
                ("DNS/256", "examples/DNS/256"),
                ("DNS/128", "examples/DNS/128"),
                ("LES/128", "examples/LES/128"),
                ("LES/64", "examples/LES/64"),
            ]
            
            selected_quick = st.multiselect(
                "Select directories:",
                options=[label for label, _ in quick_dirs],
                help="Select multiple directories to load at once"
            )
            
            if st.button("Load Selected", key="load_quick_multi"):
                selected_paths = [path for label, path in quick_dirs if label in selected_quick]
                valid_dirs = []
                for rel_path in selected_paths:
                    resolved = _resolve_directory_path(rel_path)
                    if resolved:
                        valid_dirs.append(str(resolved))
                
                if _load_directories_and_scan(valid_dirs):
                    st.success(f"✅ Loaded {len(valid_dirs)} directories")
                    st.rerun()
        
        else:
            # Single directory mode (original behavior)
            st.markdown("**Select your simulation output folder:**")
            
            # Allow both absolute and relative paths
            user_dir = st.text_input(
                "Directory path:",
                value="",
                help="Enter absolute path or path relative to project root (e.g., examples/DNS/256)"
            )
            
            # Quick access to common locations
            with st.expander("📂 Quick Access to Project Directories", expanded=False):
                st.markdown("**Common locations:**")
                quick_paths = [
                    ("examples/DNS/128", "examples/DNS/128"),
                    ("examples/DNS/256", "examples/DNS/256"),
                    ("examples/DNS/512", "examples/DNS/512"),
                    ("examples/LES/64", "examples/LES/64"),
                    ("examples/LES/128", "examples/LES/128"),
                    ("examples", "examples"),
                ]
                for label, path in quick_paths:
                    resolved = _resolve_directory_path(path, log_warnings=False)
                    if resolved:
                        if st.button(f"{label}", key=f"quick_{path}", width='stretch'):
                            if _load_single_directory_and_merge(path):
                                st.success(f"Loaded: {path}")
                                st.rerun()
            
            if st.button("Load Data", type="primary"):
                if _load_single_directory_and_merge(user_dir):
                    resolved = _resolve_directory_path(user_dir)
                    try:
                        rel_path = resolved.relative_to(project_root)
                        st.success(f"Data loaded from: {rel_path}")
                    except ValueError:
                        st.success(f"Data loaded from: {user_dir}")
                elif user_dir:
                    st.error(f"Directory not found: {user_dir}")
                    st.info(f"Tip: Use path relative to project root (e.g., examples/DNS/256)")
                else:
                    st.warning("Please enter a directory path.")
            
            # Optional: Browse project directories
            st.markdown("---")
            st.markdown("**Or browse project directories:**")
            
            # Find all data directories in the project
            all_dirs = []
            
            # Search in examples
            example_base = project_root / "examples"
            if example_base.exists():
                # Look for DNS subdirectories
                dns_dir = example_base / "DNS"
                if dns_dir.exists():
                    for subdir in sorted(dns_dir.iterdir()):
                        if subdir.is_dir():
                            all_dirs.append(("DNS/" + subdir.name, subdir))
                # Check LES subdirectories
                les_dir = example_base / "LES"
                if les_dir.exists():
                    for subdir in sorted(les_dir.iterdir()):
                        if subdir.is_dir():
                            all_dirs.append(("LES/" + subdir.name, subdir))
                # Also check direct subdirectories in examples
                for subdir in sorted(example_base.iterdir()):
                    if subdir.is_dir() and subdir.name not in ["DNS", "LES"]:
                        all_dirs.append((subdir.name, subdir))
            
            # Search in other common locations (optional - can be expanded)
            # More search paths can be added here if needed
            
            if all_dirs:
                # Create user-friendly display names (relative to project root)
                dir_options = {}
                for name, path in all_dirs:
                    # Get relative path from project root
                    try:
                        rel_path = path.relative_to(project_root)
                        display_name = f"{name} ({rel_path})"
                    except ValueError:
                        # If path is not relative to project root, just use name
                        display_name = name
                    dir_options[display_name] = str(path)
                
                selected_dir = st.selectbox(
                    "Select dataset directory:",
                    options=list(dir_options.keys()),
                    help="Choose a dataset directory from the project"
                )
                if st.button("Load Selected Directory", type="primary"):
                    selected_path = dir_options[selected_dir]
                    if _load_single_directory_and_merge(selected_path):
                        try:
                            rel_path = Path(selected_path).relative_to(project_root)
                            st.success(f"Data loaded from: {rel_path}")
                        except ValueError:
                            st.success(f"Data loaded from: {selected_path}")
            else:
                st.info("No data directories found. You can still load your own data using the directory path above.")
    
    # Main content area
    if st.session_state.data_loaded:
        _display_logo_or_title()
    
    if not st.session_state.data_loaded:
        # Get theme colors
        colors = _get_theme_colors()
        card_bg = colors['card_bg']
        text_color = colors['text_color']
        secondary_text = colors['secondary_text']
        border_color = colors['border_color']
        accent_color = colors['accent_color']
        
        # Hero Section
        st.markdown(f"""
        <div style='text-align: center; padding: 0.3rem 0 0.5rem 0; border-bottom: 2px solid {border_color}; margin-bottom: 0.8rem;'>
            <h1 style='font-size: 1.8rem; margin: 0; color: {text_color}; font-weight: 600;'>KI-TURB 3D</h1>
            <p style='font-size: 0.9rem; color: {secondary_text}; margin: 0.2rem 0 0 0;'>Turbulence Analysis & Visualization Suite</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main description - compact
        st.markdown(f"""
        <div style='background: {card_bg}; padding: 0.6rem; border-radius: 4px; border-left: 3px solid {accent_color}; margin-bottom: 0.8rem;'>
            <p style='margin: 0; font-size: 0.9rem; color: {text_color}; line-height: 1.5;'>
                <strong>KI-TURB 3D</strong> analyzes <strong>Lattice Boltzmann Method (LBM)</strong> <strong>Direct Numerical Simulation (DNS)</strong> and <strong>Large Eddy Simulation (LES)</strong> turbulence data. 
                Data is primarily from <strong>Multiple Relaxation Time (MRT)</strong> simulations; the tool can be used seamlessly for <strong>Single Relaxation Time (SRT)</strong>, <strong>Bhatnagar-Gross-Krook (BGK)</strong>, and <strong>Two Relaxation Time (TRT)</strong> data. 
                Core analysis: Energy spectra <em>E</em>(<em>k</em>), structure functions <em>S</em><sub><em>p</em></sub>(<em>r</em>), scaling exponents ξ<sub><em>p</em></sub> via <strong>Extended Self-Similarity (ESS)</strong>, isotropy validation, flatness factors, probability density functions (PDFs), time series statistics, energy balance, 3D visualization, and multi-simulation comparison.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Cards - Compact Grid (4 columns)
        st.markdown(f"""
        <h3 style='font-size: 1.1rem; color: {text_color}; margin: 0.5rem 0; border-bottom: 1px solid {border_color}; padding-bottom: 0.3rem;'>Analysis Capabilities</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Energy Spectra</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'><em>E</em>(<em>k</em>) with Kolmogorov scaling</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>PDFs & Statistics</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>PDFs, time series, Re numbers</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color};'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Theory & Equations</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>D3Q19, MRT matrix, equations</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Structure Functions</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'><em>S</em><sub><em>p</em></sub>(<em>r</em>) with ESS and scaling</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>3D Visualization</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>Interactive 3D volume viewer</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color};'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Multi-Simulation</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>Compare multiple DNS/LES runs</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Isotropy Validation</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>Spectral and real-space analysis</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Energy Balance</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>MRT/SRT dissipation tracking</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color};'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Report Generator</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>PDF/HTML reports</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Flatness Factors</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'><em>F</em>(<em>r</em>) and intermittency</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Time Series</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>TKE, dissipation, Re evolution</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color}; margin-bottom: 0.4rem;'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>Overview</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>Simulation metadata & stats</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 4px; background: {card_bg}; border-left: 3px solid {accent_color};'>
                <strong style='color: {text_color}; font-size: 0.9rem; display: block; margin-bottom: 0.2rem;'>AI Assistant</strong>
                <span style='color: {secondary_text}; font-size: 0.75rem; line-height: 1.3;'>Natural language interface</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Details - Compact
        st.markdown(f"""
        <h3 style='font-size: 1.1rem; color: {text_color}; margin: 0.8rem 0 0.4rem 0; border-bottom: 1px solid {border_color}; padding-bottom: 0.3rem;'>Technical Specifications</h3>
        <div style='background: {card_bg}; padding: 0.5rem; border-radius: 4px; font-size: 0.8rem; line-height: 1.6; color: {text_color};'>
            <strong>Formats:</strong> .bin, .dat, .txt, .vti, .h5, .hdf5, .csv, simulation.input | 
            <strong>Processing:</strong> Time-averaging, multi-sim grouping, error bars | 
            <strong>Visualization:</strong> Plotly, 3D viewer, Light/Dark themes, export PNG/PDF/SVG/EPS | 
            <strong>Analysis:</strong> Kolmogorov scaling, ESS, anomalous scaling, isotropy, PDFs, energy balance
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show loaded directories in main content area
        if st.session_state.multi_directory_mode and st.session_state.data_directories:
            st.success(f"✅ Loaded {len(st.session_state.data_directories)} simulation directories:")
            for i, data_dir_path in enumerate(st.session_state.data_directories, 1):
                data_dir = Path(data_dir_path)
                try:
                    rel_path = data_dir.relative_to(project_root)
                    display_path = str(rel_path)
                except ValueError:
                    display_path = str(data_dir)
                st.markdown(f"  **{i}.** `{display_path}`")
            st.info("Pages will automatically load files from all selected directories for comparison.")
        else:
            # Single directory mode
            data_dir = Path(st.session_state.data_directory)
            try:
                rel_path = data_dir.relative_to(project_root)
                display_path = str(rel_path)
            except ValueError:
                display_path = str(data_dir)
            st.success(f"✅ Data loaded from: `{display_path}`")
            st.info("Navigate through the pages using the sidebar menu.")

if __name__ == "__main__":
    main()


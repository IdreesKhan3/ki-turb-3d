"""
Turbulence Statistics Dashboard
Main Streamlit application entry point
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.theme_config import get_theme_list, get_default_theme, inject_theme_css

# Page configuration
st.set_page_config(
    page_title="IK-TURB 3D",
    page_icon="🌊",
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

def main():
    """Main application entry point"""
    
    # Apply theme CSS globally (must be first to apply to all pages)
    inject_theme_css()
    
    # Sidebar - Data Selection
    with st.sidebar:
        # Logo (if exists)
        logo_path = project_root / "logo.png"
        if logo_path.exists() and logo_path.stat().st_size > 0:
            try:
                st.image(str(logo_path), use_container_width=True)
            except Exception:
                st.title("🌊 IK-TURB 3D")
                st.caption("Turbulence Visualization & Analysis Suite")
        else:
            st.title("🌊 IK-TURB 3D")
            st.caption("Turbulence Visualization & Analysis Suite")
        st.markdown("---")
        
        # Theme Selector
        st.subheader("🎨 Theme")
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
        from utils.theme_config import get_theme
        theme_info = get_theme(selected_theme)
        st.caption(f"💡 {theme_info['description']}")
        
        st.markdown("---")
        
        st.subheader("📁 Data Selection")
        
        # Mode selection: Single or Multiple directories
        selection_mode = st.radio(
            "Selection Mode:",
            ["Single Simulation", "Multiple Simulations (Comparison)"],
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
                help="Enter absolute paths or paths relative to project root, one per line.\nExample:\nexamples/showcase/DNS/768\nexamples/showcase/DNS/512\nexamples/showcase/LES/128"
            )
            
            if st.button("Load Multiple Directories", type="primary"):
                dir_paths = [p.strip() for p in dirs_input.strip().split("\n") if p.strip()]
                valid_dirs = []
                
                for dir_path in dir_paths:
                    # Try as absolute path first
                    if Path(dir_path).exists() and Path(dir_path).is_dir():
                        valid_dirs.append(str(Path(dir_path).absolute()))
                    # Try as relative path from project root
                    else:
                        relative_path = project_root / dir_path
                        if relative_path.exists() and relative_path.is_dir():
                            valid_dirs.append(str(relative_path.absolute()))
                        else:
                            st.warning(f"Directory not found: {dir_path}")
                
                if valid_dirs:
                    st.session_state.data_directories = valid_dirs
                    # For backward compatibility, set first directory as primary
                    st.session_state.data_directory = valid_dirs[0]
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(valid_dirs)} directories")
                    for i, d in enumerate(valid_dirs, 1):
                        try:
                            rel_path = Path(d).relative_to(project_root)
                            st.caption(f"  {i}. APP/{rel_path}")
                        except ValueError:
                            st.caption(f"  {i}. {d}")
                    st.rerun()
                else:
                    st.error("No valid directories found. Please check your paths.")
            
            # Quick multi-select from common locations
            st.markdown("---")
            st.markdown("**Quick Multi-Select:**")
            quick_dirs = [
                ("DNS/768", "examples/showcase/DNS/768"),
                ("DNS/512", "examples/showcase/DNS/512"),
                ("DNS/256", "examples/showcase/DNS/256"),
                ("DNS/128", "examples/showcase/DNS/128"),
                ("LES/128", "examples/showcase/LES/128"),
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
                    full_path = project_root / rel_path
                    if full_path.exists() and full_path.is_dir():
                        valid_dirs.append(str(full_path.absolute()))
                
                if valid_dirs:
                    st.session_state.data_directories = valid_dirs
                    st.session_state.data_directory = valid_dirs[0]
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(valid_dirs)} directories")
                    st.rerun()
        
        else:
            # Single directory mode (original behavior)
            st.markdown("**Select your simulation output folder:**")
        
        # Allow both absolute and relative paths
        user_dir = st.text_input(
            "Directory path:",
            value="",
            help="Enter absolute path or path relative to project root (e.g., examples/showcase/DNS/256)"
        )
        
        # Quick access to common locations
        with st.expander("📂 Quick Access to Project Directories", expanded=False):
            st.markdown("**Common locations:**")
            quick_paths = [
                ("examples/showcase/DNS/256", "examples/showcase/DNS/256"),
                ("examples/showcase/DNS/512", "examples/showcase/DNS/512"),
                ("examples/showcase/DNS/768", "examples/showcase/DNS/768"),
                ("examples/showcase/LES/128", "examples/showcase/LES/128"),
                ("examples/showcase", "examples/showcase"),
            ]
            for label, path in quick_paths:
                full_path = project_root / path
                if full_path.exists():
                    if st.button(f"📁 {label}", key=f"quick_{path}", use_container_width=True):
                        st.session_state.data_directory = str(full_path)
                        st.session_state.data_directories = [str(full_path)]
                        st.session_state.data_loaded = True
                        st.success(f"Loaded: APP/{path}")
                        st.rerun()
        
        if st.button("Load Data", type="primary"):
            # Try as absolute path first
            if user_dir and Path(user_dir).exists() and Path(user_dir).is_dir():
                abs_path = Path(user_dir).absolute()
                st.session_state.data_directory = str(abs_path)
                st.session_state.data_directories = [str(abs_path)]  # Also set in list for compatibility
                st.session_state.data_loaded = True
                # Show relative path if within project, otherwise show input as-is
                try:
                    rel_path = abs_path.relative_to(project_root)
                    st.success(f"Data loaded from: APP/{rel_path}")
                except ValueError:
                    st.success(f"Data loaded from: {user_dir}")
            # Try as relative path from project root
            elif user_dir:
                relative_path = project_root / user_dir
                if relative_path.exists() and relative_path.is_dir():
                    st.session_state.data_directory = str(relative_path.absolute())
                    st.session_state.data_directories = [str(relative_path.absolute())]
                    st.session_state.data_loaded = True
                    st.success(f"Data loaded from: APP/{user_dir}")
                else:
                    st.error(f"Directory not found: {user_dir}")
                    st.info(f"💡 Tip: Use path relative to project root (e.g., examples/showcase/DNS/256)")
            else:
                st.warning("Please enter a directory path.")
        
        # Optional: Browse project directories
        st.markdown("---")
        st.markdown("**Or browse project directories:**")
        
        # Find all data directories in the project
        all_dirs = []
        
        # Search in examples/showcase
        example_base = project_root / "examples" / "showcase"
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
            # Also check direct subdirectories in showcase
            for subdir in sorted(example_base.iterdir()):
                if subdir.is_dir() and subdir.name not in ["DNS", "LES"]:
                    all_dirs.append((subdir.name, subdir))
        
        # Search in other common locations (optional - can be expanded)
        # You can add more search paths here if needed
        
        if all_dirs:
            # Create user-friendly display names (relative to project root)
            dir_options = {}
            for name, path in all_dirs:
                # Get relative path from project root
                try:
                    rel_path = path.relative_to(project_root)
                    display_name = f"{name} (APP/{rel_path})"
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
                st.session_state.data_directory = selected_path
                st.session_state.data_directories = [selected_path]
                st.session_state.data_loaded = True
                # Show relative path in success message
                try:
                    rel_path = Path(selected_path).relative_to(project_root)
                    st.success(f"Data loaded from: APP/{rel_path}")
                except ValueError:
                    st.success(f"Data loaded from: {selected_path}")
        else:
            st.info("No data directories found. You can still load your own data using the directory path above.")
        
        # Navigation
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        st.markdown("""
        - [Overview](#overview)
        - [Energy Spectra](#energy-spectra)
        - [Structure Functions](#structure-functions)
        - [Flatness](#flatness)
        - [LES Metrics](#les-metrics)
        - [Isotropy](#isotropy)
        - [Comparison](#comparison)
        - [Reynolds Transition](#reynolds-transition)
        - [Theory & Equations](#theory-equations)
        - [Multi-Method Support](#multi-method-support)
        """)
    
    # Main content area
    # Logo at top (if exists), otherwise text title
    logo_path = project_root / "logo.png"
    if logo_path.exists() and logo_path.stat().st_size > 0:
        try:
            st.image(str(logo_path), use_container_width=True)
        except Exception:
            st.title("🌊 IK-TURB 3D")
            st.caption("Turbulence Visualization & Analysis Suite")
    else:
        st.title("🌊 IK-TURB 3D")
        st.caption("Turbulence Visualization & Analysis Suite")
    
    if not st.session_state.data_loaded:
        st.info("👈 Please select a data directory from the sidebar to begin analysis.")
        st.markdown("""
        ### Welcome!
        
        This dashboard allows you to analyze LBM-based DNS/LES turbulence simulation data.
        **Primary focus:** MRT (Multiple Relaxation Time) | **Also supports:** SRT (BGK)
        
        **Features:**
        - Interactive energy spectra visualization
        - Structure functions and ESS analysis
        - Flatness factors
        - Isotropy validation (spectral and real-space)
        - LES metrics
        - Other turbulence statistics and energy balance analysis
        - Multi-simulation comparison
        - Multi-method support (MRT and SRT).
        
        **Getting Started:**
        1. Select your simulation output folder from the sidebar
        2. Or try the example data to see the dashboard in action
        3. Navigate through the pages using the sidebar
        """)
    else:
        # Show loaded directories in main content area
        if st.session_state.multi_directory_mode and st.session_state.data_directories:
            st.success(f"✅ Loaded {len(st.session_state.data_directories)} simulation directories:")
            for i, data_dir_path in enumerate(st.session_state.data_directories, 1):
                data_dir = Path(data_dir_path)
                try:
                    rel_path = data_dir.relative_to(project_root)
                    display_path = f"APP/{rel_path}"
                except ValueError:
                    display_path = str(data_dir)
                st.markdown(f"  **{i}.** `{display_path}`")
            st.info("💡 Pages will automatically load files from all selected directories for comparison.")
        else:
            # Single directory mode
            data_dir = Path(st.session_state.data_directory)
            try:
                rel_path = data_dir.relative_to(project_root)
                display_path = f"APP/{rel_path}"
            except ValueError:
                display_path = str(data_dir)
            st.success(f"✅ Data loaded from: `{display_path}`")
            st.info("Navigate through the pages using the sidebar menu.")

if __name__ == "__main__":
    main()


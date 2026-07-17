"""Other turbulence statistics page (Streamlit).

Plots turbulence_stats* and energy-balance residuals from eps_real_validation*.
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import apply_theme_to_plot_style, inject_theme_css
from utils.plot_style import default_plot_style

from pages.OtherTurbStats import (
    init_session_state,
    load_all_data,
    render_custom_plot_section,
    render_tables_section,
)

st.set_page_config(page_icon="⚫")


def main():
    inject_theme_css()
    st.title("Other Turbulence Stats")

    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]

    if data_dirs and len(data_dirs) > 1:
        st.info(f"📁 **Multiple simulations loaded:** {len(data_dirs)} directories")
        with st.expander("View loaded directories", expanded=False):
            for i, data_dir_path in enumerate(data_dirs, 1):
                data_dir_obj = Path(data_dir_path)
                try:
                    rel_path = data_dir_obj.relative_to(project_root)
                    st.markdown(f"**{i}.** `{rel_path}`")
                except ValueError:
                    st.markdown(f"**{i}.** `{data_dir_path}`")
        st.markdown("---")

    current_theme = st.session_state.get("theme", "Light Scientific")
    if "plot_style" not in st.session_state:
        st.session_state.plot_style = default_plot_style()
    st.session_state.plot_style = apply_theme_to_plot_style(
        st.session_state.plot_style,
        current_theme,
    )

    init_session_state()
    result = load_all_data()
    if result is None:
        return

    data_dir, data_dirs, all_dataframes, table_data = result

    render_custom_plot_section(data_dir, all_dataframes)
    st.markdown("---")
    render_tables_section(data_dirs, table_data)


if __name__ == "__main__":
    main()

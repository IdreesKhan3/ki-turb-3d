"""
Theme configuration system for scientific visualization
Provides predefined themes optimized for different use cases
"""

import plotly.colors as pc


THEMES = {
    "Light Scientific": {
        "name": "Light Scientific",
        "description": "Default light theme for scientific analysis",
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "font_color": "#000000",
        "grid_color": "#B0B0B0",
        "grid_opacity": 0.6,
        "minor_grid_color": "#D0D0D0",
        "minor_grid_opacity": 0.45,
        "palette": "Plotly",
        "custom_colors": ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                          "#8c564b", "#e377c2", "#7f7f7f"],
        "template": "plotly_white",
        "font_family": "Arial",
        "font_size": 14,
        "title_size": 16,
        "legend_size": 12,
        "tick_font_size": 12,
        "axis_title_size": 14,
    },
    
    "Dark Scientific": {
        "name": "Dark Scientific",
        "description": "VS Code-style dark theme for extended viewing",
        "plot_bgcolor": "#1e1e1e",
        "paper_bgcolor": "#1e1e1e",
        "font_color": "#d4d4d4",
        "grid_color": "#3e3e42",
        "grid_opacity": 0.6,
        "minor_grid_color": "#2d2d30",
        "minor_grid_opacity": 0.4,
        "palette": "Custom",
        "custom_colors": ["#4ec9b0", "#569cd6", "#dcdcaa", "#ce9178",
                          "#c586c0", "#d7ba7d", "#9cdcfe"],
        "template": "plotly_dark",
        "font_family": "Consolas, 'Courier New', monospace",
        "font_size": 14,
        "title_size": 16,
        "legend_size": 12,
        "tick_font_size": 12,
        "axis_title_size": 14,
    },
    
    "Publication": {
        "name": "Publication",
        "description": "Nature/Science style - publication-ready",
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "font_color": "#000000",
        "grid_color": "#CCCCCC",
        "grid_opacity": 0.4,
        "minor_grid_color": "#E5E5E5",
        "minor_grid_opacity": 0.3,
        "palette": "Custom",
        "custom_colors": ["#000000", "#E64A19", "#1976D2", "#388E3C",
                          "#F57C00", "#7B1FA2", "#C2185B"],
        "template": "simple_white",
        "font_family": "Arial",
        "font_size": 12,
        "title_size": 14,
        "legend_size": 11,
        "tick_font_size": 11,
        "axis_title_size": 12,
    },
    
    "Presentation": {
        "name": "Presentation",
        "description": "High contrast for presentations and projectors",
        "plot_bgcolor": "#000000",
        "paper_bgcolor": "#000000",
        "font_color": "#FFFFFF",
        "grid_color": "#666666",
        "grid_opacity": 0.6,
        "minor_grid_color": "#444444",
        "minor_grid_opacity": 0.4,
        "palette": "Bold",
        "custom_colors": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00",
                          "#FF00FF", "#00FFFF", "#FFFFFF"],
        "template": "plotly_dark",
        "font_family": "Arial",
        "font_size": 16,
        "title_size": 20,
        "legend_size": 14,
        "tick_font_size": 14,
        "axis_title_size": 16,
    },
    
    "Colorblind Friendly": {
        "name": "Colorblind Friendly",
        "description": "Accessible colors for color vision deficiencies",
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "font_color": "#000000",
        "grid_color": "#B0B0B0",
        "grid_opacity": 0.6,
        "minor_grid_color": "#D0D0D0",
        "minor_grid_opacity": 0.45,
        "palette": "Custom",
        "custom_colors": ["#006BA4", "#FF800E", "#ABABAB", "#595959",
                          "#5F9ED1", "#C85200", "#898989"],
        "template": "plotly_white",
        "font_family": "Arial",
        "font_size": 14,
        "title_size": 16,
        "legend_size": 12,
        "tick_font_size": 12,
        "axis_title_size": 14,
    },
    
    "High Contrast": {
        "name": "High Contrast",
        "description": "Maximum contrast for visibility",
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "font_color": "#000000",
        "grid_color": "#000000",
        "grid_opacity": 0.8,
        "minor_grid_color": "#666666",
        "minor_grid_opacity": 0.6,
        "palette": "Custom",
        "custom_colors": ["#000000", "#FF0000", "#0000FF", "#008000",
                          "#FF8000", "#800080", "#FF00FF"],
        "template": "simple_white",
        "font_family": "Arial",
        "font_size": 15,
        "title_size": 18,
        "legend_size": 13,
        "tick_font_size": 13,
        "axis_title_size": 15,
    },
}


def get_theme(theme_name: str) -> dict:
    """
    Get theme configuration by name
    
    Args:
        theme_name: Name of the theme
        
    Returns:
        Dictionary with theme configuration
    """
    return THEMES.get(theme_name, THEMES["Light Scientific"]).copy()


def apply_theme_to_plot_style(plot_style: dict, theme_name: str) -> dict:
    """
    Apply theme settings to an existing plot style dictionary
    
    Args:
        plot_style: Existing plot style dictionary
        theme_name: Name of theme to apply
        
    Returns:
        Updated plot style dictionary
    """
    theme = get_theme(theme_name)
    
    # Update plot style with theme settings
    updated_style = plot_style.copy()
    
    # Backgrounds
    updated_style["plot_bgcolor"] = theme["plot_bgcolor"]
    updated_style["paper_bgcolor"] = theme["paper_bgcolor"]
    
    # Grid
    updated_style["grid_color"] = theme["grid_color"]
    updated_style["grid_opacity"] = theme["grid_opacity"]
    updated_style["minor_grid_color"] = theme["minor_grid_color"]
    updated_style["minor_grid_opacity"] = theme["minor_grid_opacity"]
    
    # Colors
    updated_style["palette"] = theme["palette"]
    updated_style["custom_colors"] = theme["custom_colors"].copy()
    
    # Template
    updated_style["template"] = theme["template"]
    
    # Fonts
    updated_style["font_family"] = theme["font_family"]
    updated_style["font_size"] = theme["font_size"]
    updated_style["title_size"] = theme["title_size"]
    updated_style["legend_size"] = theme["legend_size"]
    updated_style["tick_font_size"] = theme["tick_font_size"]
    updated_style["axis_title_size"] = theme["axis_title_size"]
    
    return updated_style


def get_theme_list() -> list:
    """Get list of available theme names"""
    return list(THEMES.keys())


def get_default_theme() -> str:
    """Get the default theme name"""
    return "Light Scientific"


def _get_palette(ps):
    """
    Get color palette from plot style (compatible with existing code)
    
    Args:
        ps: Plot style dictionary
        
    Returns:
        List of colors
    """
    if ps.get("palette") == "Custom":
        cols = ps.get("custom_colors", [])
        return cols if cols else pc.qualitative.Plotly
    
    mapping = {
        "Plotly": pc.qualitative.Plotly,
        "D3": pc.qualitative.D3,
        "G10": pc.qualitative.G10,
        "T10": pc.qualitative.T10,
        "Dark2": pc.qualitative.Dark2,
        "Set1": pc.qualitative.Set1,
        "Set2": pc.qualitative.Set2,
        "Pastel1": pc.qualitative.Pastel1,
        "Bold": pc.qualitative.Bold,
        "Prism": pc.qualitative.Prism,
    }
    return mapping.get(ps.get("palette", "Plotly"), pc.qualitative.Plotly)


def inject_theme_css(theme_name: str = None):
    """
    Inject CSS for theme styling across all pages
    
    Args:
        theme_name: Name of theme (if None, uses session state)
    """
    import streamlit as st
    
    if theme_name is None:
        theme_name = st.session_state.get("theme", "Light Scientific")
    
    theme_info = get_theme(theme_name)
    
    # Only inject CSS for dark themes
    if "Dark" in theme_name or "Presentation" in theme_name:
        bg_color = theme_info['paper_bgcolor']
        sidebar_color = "#252526"
        text_color = theme_info.get('font_color', '#d4d4d4')
        border_color = "#3e3e42"
        input_bg = "#3c3c3c"
        header_bg = "#2d2d30"
        
        st.markdown(f"""
        <style>
        /* Main app background */
        .stApp {{
            background-color: {bg_color} !important;
        }}
        
        /* Streamlit header/menu bar */
        header[data-testid="stHeader"] {{
            background-color: {header_bg} !important;
        }}
        
        header[data-testid="stHeader"] > div {{
            background-color: {header_bg} !important;
        }}
        
        /* Deploy button and menu */
        .stDeployButton {{
            background-color: {header_bg} !important;
        }}
        
        .stDeployButton > button {{
            background-color: {header_bg} !important;
            color: {text_color} !important;
        }}
        
        /* All buttons in header */
        header button {{
            background-color: {header_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Menu button icons */
        header button svg {{
            fill: {text_color} !important;
        }}
        
        /* Main content area */
        .main {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_color} !important;
        }}
        
        [data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}
        
        /* Headers and text */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_color} !important;
        }}
        
        p, div, span, label {{
            color: {text_color} !important;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border-color: {border_color} !important;
        }}
        
        .stSelectbox > div > div > select {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border-color: {border_color} !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: #0e639c !important;
            color: white !important;
            border-color: {border_color} !important;
        }}
        
        .stButton > button:hover {{
            background-color: #1177bb !important;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {sidebar_color} !important;
            color: {text_color} !important;
        }}
        
        /* Dataframes */
        .dataframe {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Info/Warning boxes */
        .stAlert {{
            background-color: {input_bg} !important;
            border-color: {border_color} !important;
        }}
        
        /* Markdown text */
        .stMarkdown {{
            color: {text_color} !important;
        }}
        
        /* Caption text */
        .stCaption {{
            color: {text_color} !important;
            opacity: 0.8;
        }}
        
        /* Menu dropdown */
        div[data-baseweb="popover"], div[data-baseweb="menu"] {{
            background-color: {input_bg} !important;
        }}
        
        div[data-baseweb="menu"] li {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        div[data-baseweb="menu"] li:hover {{
            background-color: {border_color} !important;
        }}
        </style>
        """, unsafe_allow_html=True)


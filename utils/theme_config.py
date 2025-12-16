"""
Theme configuration system for scientific visualization
Provides Light Scientific and Dark Scientific themes
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
        "axis_line_color": "#000000",  # Black borders for light theme
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
        "axis_line_color": "#d4d4d4",  # Light gray borders for dark theme (visible on dark background)
        "palette": "Custom",
        "custom_colors": ["#4ec9b0", "#569cd6", "#dcdcaa", "#ce9178",
                          "#c586c0", "#d7ba7d", "#9cdcfe"],
        "template": "plotly_dark",
        "font_family": "Courier New",
        "font_size": 14,
        "title_size": 16,
        "legend_size": 12,
        "tick_font_size": 12,
        "axis_title_size": 14,
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
    
    # Axis borders (spines)
    updated_style["axis_line_color"] = theme["axis_line_color"]
    
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
    
    # Font color
    updated_style["font_color"] = theme["font_color"]
    
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


def template_selector(ps):
    """
    Template selector with automatic background color updates.
    
    Args:
        ps: Plot style dictionary (will be modified in place)
        
    Returns:
        Selected template name
    """
    import streamlit as st
    
    templates = ["plotly_white", "simple_white", "plotly_dark"]
    old_template = ps.get("template", "plotly_white")
    ps["template"] = st.selectbox("Template", templates,
                                  index=templates.index(old_template))
    
    # Auto-update backgrounds when template changes
    if ps["template"] != old_template:
        if ps["template"] == "plotly_dark":
            ps["plot_bgcolor"] = "#1e1e1e"
            ps["paper_bgcolor"] = "#1e1e1e"
        else:  # plotly_white or simple_white
            ps["plot_bgcolor"] = "#FFFFFF"
            ps["paper_bgcolor"] = "#FFFFFF"
    
    return ps["template"]


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
    if "Dark" in theme_name:
        bg_color = theme_info['paper_bgcolor']
        sidebar_color = "#252526"
        text_color = "#e8e8e8"
        bright_text = "#ffffff"
        border_color = "#3e3e42"
        input_bg = "#3c3c3c"
        header_bg = "#2d2d30"
        container_bg = "#2d2d30"
        
        st.markdown(f"""
        <style>
        /* CSS Variables for theme (accessible to custom components) */
        :root {{
            --background-color: {bg_color};
            --container-bg: {container_bg};
            --app-bg: {bg_color};
            --app-container-bg: {container_bg};
            --app-input-bg: {input_bg};
            --app-border: {border_color};
            --app-text: {text_color};
            --app-text-strong: {bright_text};
            --app-sidebar-bg: {sidebar_color};
            --app-header-bg: {header_bg};
        }}
        
        /* Main app background */
        .stApp {{
            background-color: {bg_color} !important;
        }}
        
        /* Sticky input bar area (stForm) - aggressive targeting */
        [data-testid="stForm"],
        [data-testid="stForm"] *,
        [data-testid="stForm"] * * {{
            background: var(--background-color) !important;
        }}
        
        [data-testid="stForm"] {{
            background: var(--background-color) !important;
        }}
        
        /* All possible Streamlit containers around the form */
        [data-testid="stForm"] .element-container,
        [data-testid="stForm"] .block-container,
        [data-testid="stForm"] [data-testid="stVerticalBlock"],
        [data-testid="stForm"] [data-testid="stHorizontalBlock"],
        [data-testid="stForm"] [data-testid="stColumn"],
        [data-testid="stForm"] > div,
        [data-testid="stForm"] > div > div,
        [data-testid="stForm"] > div > div > div,
        /* Target columns that contain the input */
        [data-testid="column"]:has([data-testid="stForm"]),
        [data-testid="column"]:has(iframe[title*="multimodal"]),
        [data-testid="column"]:has(iframe[title*="chat_input"]),
        /* Target any div that might wrap the form */
        div:has([data-testid="stForm"]),
        div:has(iframe[title*="multimodal"]),
        div:has(iframe[title*="chat_input"]) {{
            background: var(--background-color) !important;
        }}
        
        /* Remove card styling - blend with page */
        [data-testid="stForm"] > div {{
            background: var(--background-color) !important;
            border-radius: 0 !important;
            padding: 0 !important;
        }}
        
        /* Override any white backgrounds aggressively */
        [data-testid="stForm"] [style*="background-color: white"],
        [data-testid="stForm"] [style*="background-color: #fff"],
        [data-testid="stForm"] [style*="background-color: #ffffff"],
        [data-testid="stForm"] * [style*="background-color: white"],
        [data-testid="stForm"] * [style*="background-color: #fff"],
        [data-testid="stForm"] * [style*="background-color: #ffffff"] {{
            background: var(--background-color) !important;
        }}
        
        /* Scrollbar - always visible */
        * {{
            scrollbar-width: thin;
            scrollbar-color: #5a5a5a {container_bg};
        }}
        
        *::-webkit-scrollbar {{
            width: 14px !important;
            height: 14px !important;
            display: block !important;
        }}
        
        *::-webkit-scrollbar-track {{
            background: {container_bg} !important;
            display: block !important;
        }}
        
        *::-webkit-scrollbar-thumb {{
            background: #5a5a5a !important;
            border-radius: 7px !important;
            border: 2px solid {container_bg} !important;
            display: block !important;
        }}
        
        *::-webkit-scrollbar-thumb:hover {{
            background: #6a6a6a !important;
        }}
        
        /* Main app scrollbar */
        .stApp::-webkit-scrollbar,
        .main::-webkit-scrollbar,
        [data-testid="stAppViewContainer"]::-webkit-scrollbar {{
            width: 14px !important;
            display: block !important;
        }}
        
        .stApp::-webkit-scrollbar-thumb,
        .main::-webkit-scrollbar-thumb,
        [data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb {{
            background: #5a5a5a !important;
            display: block !important;
        }}
        
        /* Sidebar scrollbar */
        [data-testid="stSidebar"]::-webkit-scrollbar {{
            width: 14px !important;
            display: block !important;
        }}
        
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {{
            background: #5a5a5a !important;
            display: block !important;
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
            color: {bright_text} !important;
        }}
        
        /* All buttons in header */
        header button {{
            background-color: {header_bg} !important;
            color: {bright_text} !important;
        }}
        
        /* Menu button icons */
        header button svg {{
            fill: {bright_text} !important;
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
            color: {bright_text} !important;
        }}
        
        p, div, span, label {{
            color: {text_color} !important;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
            caret-color: {bright_text} !important;
        }}
        
        /* Placeholder text for input fields */
        .stTextInput > div > div > input::placeholder {{
            color: {bright_text} !important;
            opacity: 0.7 !important;
        }}
        
        .stSelectbox > div > div > select {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
        }}
        
        /* Selectbox container and BaseWeb components */
        .stSelectbox > div {{
            background-color: {input_bg} !important;
        }}
        
        .stSelectbox > div > div {{
            background-color: {input_bg} !important;
        }}
        
        .stSelectbox [data-baseweb="select"] {{
            background-color: {input_bg} !important;
        }}
        
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stSelectbox [data-baseweb="select"] > div > div {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stSelectbox [data-baseweb="select"] span {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stSelectbox [data-baseweb="select"] input {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        /* Ensure all selectbox inner elements have dark background (except labels) */
        .stSelectbox > div > div > *:not(label) {{
            background-color: {input_bg} !important;
        }}
        
        .stSelectbox select {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stSelectbox label, .stTextInput label {{
            color: {text_color} !important;
            background-color: transparent !important;
        }}
        
        /* Number input - comprehensive styling */
        .stNumberInput {{
            background-color: transparent !important;
        }}
        
        .stNumberInput > div {{
            background-color: {input_bg} !important;
        }}
        
        .stNumberInput > div > div {{
            background-color: {input_bg} !important;
        }}
        
        .stNumberInput > div > div > input {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
            caret-color: {bright_text} !important;
        }}
        
        .stNumberInput input {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
        }}
        
        /* BaseWeb number input components */
        .stNumberInput [data-baseweb="input"] {{
            background-color: {input_bg} !important;
        }}
        
        .stNumberInput [data-baseweb="input"] > div {{
            background-color: {input_bg} !important;
        }}
        
        .stNumberInput [data-baseweb="input"] input {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        /* Number input increment/decrement buttons */
        .stNumberInput button {{
            background-color: {container_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
        }}
        
        .stNumberInput button:hover {{
            background-color: {border_color} !important;
        }}
        
        .stNumberInput button svg {{
            fill: {bright_text} !important;
        }}
        
        /* All nested elements in number input */
        .stNumberInput * {{
            color: {bright_text} !important;
        }}
        
        .stNumberInput label {{
            color: {text_color} !important;
            background-color: transparent !important;
        }}
        
        /* Override any white backgrounds in number input containers */
        .stNumberInput [style*="background-color: white"],
        .stNumberInput [style*="background-color: #fff"],
        .stNumberInput [style*="background-color: #ffffff"] {{
            background-color: {input_bg} !important;
        }}
        
        /* Text area */
        .stTextArea > div > div > textarea,
        .stTextArea textarea,
        textarea[data-baseweb="textarea"] {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
            caret-color: {bright_text} !important;
        }}
        
        /* Text area - disabled state - multiple specific selectors with highest priority */
        .stTextArea > div > div > textarea:disabled,
        .stTextArea textarea:disabled,
        textarea[data-baseweb="textarea"]:disabled,
        .stTextArea > div > div[data-baseweb="textarea"] textarea:disabled,
        div[data-baseweb="textarea"] textarea:disabled,
        textarea:disabled,
        .stTextArea > div > div > textarea[disabled="true"],
        .stTextArea textarea[disabled="true"],
        textarea[disabled="true"] {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            -webkit-text-fill-color: {bright_text} !important;
            opacity: 1 !important;
        }}
        
        /* Force white text in disabled textarea content - highest specificity */
        .stTextArea > div > div > textarea[disabled],
        .stTextArea textarea[disabled],
        .stTextArea > div > div > textarea[disabled="true"],
        .stTextArea textarea[disabled="true"] {{
            color: {bright_text} !important;
            -webkit-text-fill-color: {bright_text} !important;
        }}
        
        /* Target all textarea elements within stTextArea regardless of nesting */
        .stTextArea textarea {{
            color: {bright_text} !important;
        }}
        
        .stTextArea textarea[disabled],
        .stTextArea textarea[disabled="true"] {{
            color: {bright_text} !important;
            -webkit-text-fill-color: {bright_text} !important;
        }}
        
        /* Override any inline styles that might set text color to black */
        .stTextArea textarea[style*="color"],
        .stTextArea textarea[style*="Color"],
        .stTextArea textarea[style*="COLOR"] {{
            color: {bright_text} !important;
            -webkit-text-fill-color: {bright_text} !important;
        }}
        
        /* Universal rule for all textareas in dark theme - highest priority */
        .stTextArea textarea,
        .stTextArea > div textarea,
        .stTextArea > div > div textarea,
        .stTextArea > div > div > textarea {{
            color: {bright_text} !important;
        }}
        
        /* Text area - placeholder text */
        .stTextArea > div > div > textarea::placeholder,
        .stTextArea textarea::placeholder,
        textarea[data-baseweb="textarea"]::placeholder {{
            color: {text_color} !important;
            opacity: 0.7 !important;
        }}
        
        /* Text area - focus state */
        .stTextArea > div > div > textarea:focus,
        .stTextArea textarea:focus,
        textarea[data-baseweb="textarea"]:focus {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
        }}
        
        .stTextArea label {{
            color: {text_color} !important;
        }}
        
        /* Override any white backgrounds in text area containers */
        .stTextArea [style*="background-color: white"],
        .stTextArea [style*="background-color: #fff"],
        .stTextArea [style*="background-color: #ffffff"] {{
            background-color: {input_bg} !important;
        }}
        
        /* Multi-select */
        .stMultiSelect > div {{
            background-color: {input_bg} !important;
        }}
        
        .stMultiSelect > div > div {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stMultiSelect [data-baseweb="select"] {{
            background-color: {input_bg} !important;
        }}
        
        .stMultiSelect [data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stMultiSelect [data-baseweb="select"] > div > div {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stMultiSelect [data-baseweb="select"] input {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stMultiSelect [data-baseweb="select"] span {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stMultiSelect label {{
            color: {text_color} !important;
            background-color: transparent !important;
        }}
        
        /* Override white backgrounds in multiselect */
        .stMultiSelect [style*="background-color: white"],
        .stMultiSelect [style*="background-color: #fff"],
        .stMultiSelect [style*="background-color: #ffffff"] {{
            background-color: {input_bg} !important;
        }}
        
        /* Slider */
        .stSlider label {{
            color: {text_color} !important;
        }}
        
        .stSlider > div > div > div {{
            color: {text_color} !important;
        }}
        
        /* Checkbox */
        .stCheckbox label {{
            color: {text_color} !important;
        }}
        
        /* Radio buttons */
        .stRadio label {{
            color: {text_color} !important;
        }}
        
        .stRadio > div > label {{
            color: {text_color} !important;
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
            background-color: {container_bg} !important;
            color: {bright_text} !important;
        }}
        
        .streamlit-expanderContent {{
            background-color: {container_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Expander content area and all nested elements */
        .streamlit-expanderContent > div {{
            background-color: {container_bg} !important;
        }}
        
        /* All divs and containers inside expander content */
        .streamlit-expanderContent div,
        .streamlit-expanderContent section {{
            background-color: {container_bg} !important;
        }}
        
        /* Main content wrapper inside expander */
        .streamlit-expanderContent > div > div,
        .streamlit-expanderContent > div > div > div {{
            background-color: {container_bg} !important;
        }}
        
        /* All container divs inside expander */
        .streamlit-expanderContent div[class*="container"],
        .streamlit-expanderContent div[class*="element"],
        .streamlit-expanderContent div[class*="block"] {{
            background-color: {container_bg} !important;
        }}
        
        /* Streamlit widget containers inside expander */
        .streamlit-expanderContent [class^="st"] > div,
        .streamlit-expanderContent [class^="st"] > div > div {{
            background-color: {container_bg} !important;
        }}
        
        /* Catch-all: all divs inside expander content */
        .streamlit-expanderContent div {{
            background-color: {container_bg} !important;
        }}
        
        /* Override any white backgrounds */
        .streamlit-expanderContent [style*="background-color: white"],
        .streamlit-expanderContent [style*="background-color: #fff"],
        .streamlit-expanderContent [style*="background-color: #ffffff"] {{
            background-color: {container_bg} !important;
        }}
        
        /* Ensure main app background doesn't show through */
        .streamlit-expanderContent {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Force dark background on all states (default, hover, focus) */
        .streamlit-expanderContent,
        .streamlit-expanderContent:hover,
        .streamlit-expanderContent:focus,
        .streamlit-expanderContent:focus-within {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Force dark background on all nested divs in all states */
        .streamlit-expanderContent div,
        .streamlit-expanderContent div:hover,
        .streamlit-expanderContent div:focus,
        .streamlit-expanderContent div:focus-within {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Override any Streamlit default white backgrounds */
        .streamlit-expanderContent [style*="background"],
        .streamlit-expanderContent [style*="background-color"] {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Target the actual content area more aggressively */
        [data-testid="stExpander"] .streamlit-expanderContent,
        [data-testid="stExpander"] .streamlit-expanderContent * {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Ensure text elements maintain proper color */
        .streamlit-expanderContent p,
        .streamlit-expanderContent span,
        .streamlit-expanderContent li,
        .streamlit-expanderContent h1,
        .streamlit-expanderContent h2,
        .streamlit-expanderContent h3,
        .streamlit-expanderContent h4,
        .streamlit-expanderContent h5,
        .streamlit-expanderContent h6 {{
            color: {text_color} !important;
        }}
        
        /* Ensure expander container itself has dark background */
        [data-testid="stExpander"] {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        [data-testid="stExpander"]:hover,
        [data-testid="stExpander"]:focus,
        [data-testid="stExpander"]:focus-within {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        [data-testid="stExpander"] > div {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        [data-testid="stExpander"] > div:hover,
        [data-testid="stExpander"] > div:focus {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        [data-testid="stExpander"] > div > div {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        [data-testid="stExpander"] > div > div:hover,
        [data-testid="stExpander"] > div > div:focus {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Very specific rule to override Streamlit defaults */
        .stApp [data-testid="stExpander"] .streamlit-expanderContent,
        .stApp [data-testid="stExpander"] .streamlit-expanderContent * {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Explicitly target non-hovered state to maintain dark background */
        .streamlit-expanderContent:not(:hover),
        .streamlit-expanderContent:not(:hover) *,
        [data-testid="stExpander"]:not(:hover) .streamlit-expanderContent,
        [data-testid="stExpander"]:not(:hover) .streamlit-expanderContent * {{
            background-color: {container_bg} !important;
            background: {container_bg} !important;
        }}
        
        /* Override any CSS variables that might control background */
        .streamlit-expanderContent,
        [data-testid="stExpander"] .streamlit-expanderContent {{
            --background-color: {container_bg} !important;
            background-color: var(--background-color, {container_bg}) !important;
            background: var(--background-color, {container_bg}) !important;
        }}
        
        /* Expander content nested containers - more specific */
        .streamlit-expanderContent .element-container,
        .streamlit-expanderContent .block-container,
        .streamlit-expanderContent [class*="element-container"],
        .streamlit-expanderContent [class*="block-container"] {{
            background-color: {container_bg} !important;
        }}
        
        /* All Streamlit components inside expander */
        .streamlit-expanderContent [class*="st"] {{
            background-color: {container_bg} !important;
        }}
        
        /* Markdown content inside expander */
        .streamlit-expanderContent .stMarkdown,
        .streamlit-expanderContent .stMarkdown > div {{
            background-color: {container_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Latex content inside expander */
        .streamlit-expanderContent [class*="katex"],
        .streamlit-expanderContent [class*="latex"] {{
            background-color: {container_bg} !important;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {container_bg} !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: {container_bg} !important;
            color: {text_color} !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {bright_text} !important;
        }}
        
        /* Tab content panels */
        [data-baseweb="tab-panel"] {{
            background-color: {bg_color} !important;
        }}
        
        [data-baseweb="tab-panel"] > div {{
            background-color: {bg_color} !important;
        }}
        
        /* Content containers inside tabs */
        .stTabs [data-baseweb="tab-panel"] .element-container,
        .stTabs [data-baseweb="tab-panel"] .block-container {{
            background-color: {bg_color} !important;
        }}
        
        /* Columns and containers */
        .stColumn, [data-testid="column"] {{
            background-color: transparent !important;
        }}
        
        /* Metric boxes - comprehensive styling */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: {bright_text} !important;
        }}
        
        [data-testid="stMetricDelta"] {{
            color: {text_color} !important;
        }}
        
        [data-testid="stMetricContainer"] {{
            background-color: {container_bg} !important;
            border: 1px solid {border_color} !important;
        }}
        
        [data-testid="stMetricContainer"] > div {{
            background-color: {container_bg} !important;
        }}
        
        /* Dataframes - comprehensive styling */
        .dataframe {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border: 1px solid {border_color} !important;
        }}
        
        .dataframe th {{
            background-color: {container_bg} !important;
            color: {bright_text} !important;
            border: 1px solid {border_color} !important;
        }}
        
        .dataframe td {{
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            background-color: {input_bg} !important;
        }}
        
        .dataframe tr:nth-child(even) {{
            background-color: {container_bg} !important;
        }}
        
        .dataframe tr:hover {{
            background-color: {border_color} !important;
        }}
        
        /* Styled dataframes (pandas .style) */
        [data-testid="stDataFrame"] {{
            background-color: {input_bg} !important;
        }}
        
        [data-testid="stDataFrame"] table {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        [data-testid="stDataFrame"] th {{
            background-color: {container_bg} !important;
            color: {bright_text} !important;
        }}
        
        [data-testid="stDataFrame"] td {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        /* DataFrame container */
        [data-testid="stDataFrame"] > div {{
            background-color: {input_bg} !important;
        }}
        
        /* Info/Warning/Success/Error boxes - comprehensive styling */
        .stAlert {{
            background-color: {input_bg} !important;
            border-color: {border_color} !important;
            color: {text_color} !important;
        }}
        
        .stAlert > div {{
            color: {text_color} !important;
            background-color: {input_bg} !important;
        }}
        
        .stAlert * {{
            color: {text_color} !important;
        }}
        
        /* Success boxes */
        [data-baseweb="notification"] {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        /* Info boxes inside expanders */
        .streamlit-expanderContent .stAlert {{
            background-color: {input_bg} !important;
        }}
        
        .streamlit-expanderContent .stAlert * {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        /* All alert types */
        [data-testid="stAlert"] {{
            background-color: {input_bg} !important;
            border-color: {border_color} !important;
        }}
        
        [data-testid="stAlert"] * {{
            color: {text_color} !important;
        }}
        
        /* Markdown text */
        .stMarkdown {{
            color: {text_color} !important;
        }}
        
        /* Inline code in markdown */
        .stMarkdown code {{
            background-color: #2d2d30 !important;
            color: #e8e8e8 !important;
            border: 1px solid var(--app-border) !important;
        }}
        
        /* Code blocks - target all pre/code elements in main content */
        .main pre,
        .main code,
        .main pre code,
        [data-testid="stAppViewContainer"] pre,
        [data-testid="stAppViewContainer"] code,
        [data-testid="stAppViewContainer"] pre code,
        [data-testid="stCodeBlock"],
        [data-testid="stCodeBlock"] *,
        .stCodeBlock,
        .stCodeBlock * {{
            background-color: #1e1e1e !important;
            background: #1e1e1e !important;
            color: #d4d4d4 !important;
        }}
        
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code,
        [data-testid="stCodeBlock"] pre code,
        .stCodeBlock pre,
        .stCodeBlock code,
        .stCodeBlock pre code {{
            background-color: #1e1e1e !important;
            background: #1e1e1e !important;
            color: #d4d4d4 !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #d4d4d4 !important;
        }}
        
        /* Syntax highlighting colors (VS Code dark theme) */
        div[data-testid="stCodeBlock"] .hljs-keyword,
        div[data-testid="stCodeBlock"] code .hljs-keyword,
        .stCodeBlock .hljs-keyword,
        .stCodeBlock code .hljs-keyword {{
            color: #569cd6 !important;
        }}
        
        div[data-testid="stCodeBlock"] .hljs-string,
        div[data-testid="stCodeBlock"] code .hljs-string,
        .stCodeBlock .hljs-string,
        .stCodeBlock code .hljs-string {{
            color: #ce9178 !important;
        }}
        
        div[data-testid="stCodeBlock"] .hljs-function,
        div[data-testid="stCodeBlock"] code .hljs-function,
        .stCodeBlock .hljs-function,
        .stCodeBlock code .hljs-function {{
            color: #dcdcaa !important;
        }}
        
        div[data-testid="stCodeBlock"] .hljs-comment,
        div[data-testid="stCodeBlock"] code .hljs-comment,
        .stCodeBlock .hljs-comment,
        .stCodeBlock code .hljs-comment {{
            color: #6a9955 !important;
        }}
        
        div[data-testid="stCodeBlock"] .hljs-number,
        div[data-testid="stCodeBlock"] code .hljs-number,
        .stCodeBlock .hljs-number,
        .stCodeBlock code .hljs-number {{
            color: #b5cea8 !important;
        }}
        
        div[data-testid="stCodeBlock"] .hljs-variable,
        div[data-testid="stCodeBlock"] code .hljs-variable,
        .stCodeBlock .hljs-variable,
        .stCodeBlock code .hljs-variable {{
            color: #9cdcfe !important;
        }}
        
        div[data-testid="stCodeBlock"] .hljs-built_in,
        div[data-testid="stCodeBlock"] code .hljs-built_in,
        .stCodeBlock .hljs-built_in,
        .stCodeBlock code .hljs-built_in {{
            color: #4ec9b0 !important;
        }}
        
        /* Code blocks in wrapper containers */
        .element-container div[data-testid="stCodeBlock"],
        .block-container div[data-testid="stCodeBlock"],
        [data-testid="stVerticalBlock"] div[data-testid="stCodeBlock"],
        [data-testid="stHorizontalBlock"] div[data-testid="stCodeBlock"],
        .element-container .stCodeBlock,
        .block-container .stCodeBlock {{
            background-color: #1e1e1e !important;
        }}
        
        .element-container div[data-testid="stCodeBlock"] *,
        .block-container div[data-testid="stCodeBlock"] *,
        .element-container .stCodeBlock *,
        .block-container .stCodeBlock * {{
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
        }}
        
        /* Code blocks in chat messages */
        .stChatMessage div[data-testid="stCodeBlock"],
        .stChatMessage .stCodeBlock,
        .stChatMessage code {{
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
        }}
        
        .stChatMessage pre {{
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
        }}
        
        /* Universal code block override - highest priority */
        .stApp div[data-testid="stCodeBlock"],
        .stApp .stCodeBlock,
        .main div[data-testid="stCodeBlock"],
        .main .stCodeBlock,
        [data-testid="stAppViewContainer"] div[data-testid="stCodeBlock"],
        [data-testid="stAppViewContainer"] .stCodeBlock {{
            background-color: #1e1e1e !important;
        }}
        
        .stApp div[data-testid="stCodeBlock"] *,
        .stApp .stCodeBlock *,
        .main div[data-testid="stCodeBlock"] *,
        .main .stCodeBlock * {{
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
        }}
        
        /* Caption text */
        .stCaption {{
            color: {text_color} !important;
        }}
        
        /* Tooltips */
        [data-testid="stTooltip"] {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        /* Menu dropdown and popovers */
        div[data-baseweb="popover"] {{
            background-color: {input_bg} !important;
        }}
        
        div[data-baseweb="popover"] * {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        div[data-baseweb="menu"] {{
            background-color: {input_bg} !important;
        }}
        
        div[data-baseweb="menu"] ul {{
            background-color: {input_bg} !important;
        }}
        
        div[data-baseweb="menu"] li {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        div[data-baseweb="menu"] li > div {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        div[data-baseweb="menu"] li > div > span {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        div[data-baseweb="menu"] li:hover {{
            background-color: {border_color} !important;
        }}
        
        div[data-baseweb="menu"] li:hover > div {{
            background-color: {border_color} !important;
            color: {bright_text} !important;
        }}
        
        /* Streamlit menu items */
        [data-baseweb="menu"] [role="menuitem"] {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        [data-baseweb="menu"] [role="menuitem"]:hover {{
            background-color: {border_color} !important;
            color: {bright_text} !important;
        }}
        
        /* All popover content */
        [data-baseweb="popover"] [data-baseweb="menu"] {{
            background-color: {input_bg} !important;
        }}
        
        [data-baseweb="popover"] [data-baseweb="menu"] * {{
            color: {bright_text} !important;
        }}
        
        /* Streamlit's three-dot menu dropdown */
        header [data-baseweb="popover"] {{
            background-color: {input_bg} !important;
        }}
        
        header [data-baseweb="popover"] * {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        header [data-baseweb="popover"] li {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        header [data-baseweb="popover"] li:hover {{
            background-color: {border_color} !important;
            color: {bright_text} !important;
        }}
        
        /* Generic popup/modal backgrounds */
        [role="dialog"], [role="menu"], [role="listbox"] {{
            background-color: {input_bg} !important;
        }}
        
        [role="dialog"] *, [role="menu"] *, [role="listbox"] * {{
            color: {bright_text} !important;
        }}
        
        /* BaseWeb popover shadow container */
        div[data-baseweb="popover"] > div {{
            background-color: {input_bg} !important;
        }}
        
        /* Any nested spans or text in menu items */
        div[data-baseweb="menu"] span, 
        [data-baseweb="popover"] span,
        [data-baseweb="menu"] a {{
            color: {bright_text} !important;
        }}
        
        /* File uploader */
        .stFileUploader label {{
            color: {text_color} !important;
        }}
        
        /* Progress bar */
        .stProgress > div > div > div {{
            background-color: {border_color} !important;
        }}
        
        /* Spinner - comprehensive styling */
        .stSpinner > div {{
            border-color: {text_color} !important;
        }}
        
        .stSpinner {{
            background-color: transparent !important;
        }}
        
        /* Download buttons */
        .stDownloadButton > button {{
            background-color: #0e639c !important;
            color: white !important;
            border-color: {border_color} !important;
        }}
        
        .stDownloadButton > button:hover {{
            background-color: #1177bb !important;
        }}
        
        /* LaTeX rendering */
        .katex {{
            color: {bright_text} !important;
        }}
        
        .katex * {{
            color: {bright_text} !important;
        }}
        
        /* LaTeX in markdown */
        .stMarkdown .katex,
        .stMarkdown .katex * {{
            color: {bright_text} !important;
        }}
        
        /* LaTeX in expanders */
        .streamlit-expanderContent .katex,
        .streamlit-expanderContent .katex * {{
            color: {bright_text} !important;
        }}
        
        /* Help text and tooltips */
        [data-testid="stTooltip"] {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border: 1px solid {border_color} !important;
        }}
        
        [data-testid="stTooltip"] * {{
            color: {bright_text} !important;
        }}
        
        /* Help icon tooltips */
        [data-baseweb="tooltip"] {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        [data-baseweb="tooltip"] * {{
            color: {bright_text} !important;
        }}
        
        /* All nested text elements */
        .element-container, .block-container {{
            color: {text_color} !important;
        }}
        
        .element-container * {{
            color: inherit !important;
        }}
        
        /* Catch-all for any popup/modal elements */
        [class*="popover"], [class*="dropdown"], [class*="menu"] {{
            background-color: {input_bg} !important;
        }}
        
        [class*="popover"] *, [class*="dropdown"] *, [class*="menu"] * {{
            color: {bright_text} !important;
        }}
        
        /* BaseWeb specific menu styling */
        ul[role="menu"], ul[role="listbox"] {{
            background-color: {input_bg} !important;
        }}
        
        ul[role="menu"] li, ul[role="listbox"] li {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        ul[role="menu"] li:hover, ul[role="listbox"] li:hover {{
            background-color: {border_color} !important;
            color: {bright_text} !important;
        }}
        
        /* Cursor color for all input fields */
        input[type="text"], 
        input[type="number"], 
        input[type="email"], 
        input[type="password"], 
        input[type="search"], 
        input[type="tel"], 
        input[type="url"],
        textarea {{
            caret-color: {bright_text} !important;
        }}
        
        /* Force st_chat_input_multimodal to render dark (component is inside an iframe) */
        div[data-testid="stIFrame"] iframe[title*="chat_input_multimodal"],
        div[data-testid="stIFrame"] iframe[title*="st_chat_input_multimodal"],
        div[data-testid="stIFrame"] iframe[src*="st_chat_input_multimodal"],
        div[data-testid="stIFrame"] iframe[title*="multimodal"],
        iframe[title*="chat_input_multimodal"],
        iframe[title*="st_chat_input_multimodal"],
        iframe[src*="st_chat_input_multimodal"] {{
            filter: invert(1) hue-rotate(180deg) !important;
            border-radius: 8px !important;
        }}
        
        /* Dark surround for the multimodal component container - match page background */
        div[data-testid="stIFrame"] {{
            background: {bg_color} !important;
            padding: 0 !important;
            border-radius: 0 !important;
        }}
        
        div[data-testid="stIFrame"] > div {{
            background: transparent !important;
        }}
        
        div[data-testid="stIFrame"] iframe {{
            background: transparent !important;
            border-radius: 0 !important;
        }}
        
        /* Target the parent element-container - match page background */
        div.element-container:has(div[data-testid="stIFrame"] iframe[title*="chat_input_multimodal"]),
        div.element-container:has(div[data-testid="stIFrame"] iframe[src*="st_chat_input_multimodal"]),
        div.element-container:has(div[data-testid="stIFrame"] iframe[title*="multimodal"]) {{
            background: {bg_color} !important;
            padding: 0 !important;
            border-radius: 0 !important;
        }}
        
        /* Ensure main app canvas stays dark */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background: {bg_color} !important;
        }}
        </style>
        """, unsafe_allow_html=True)


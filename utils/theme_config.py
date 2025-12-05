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
        
        /* Number input */
        .stNumberInput > div > div > input {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
            caret-color: {bright_text} !important;
        }}
        
        .stNumberInput label {{
            color: {text_color} !important;
        }}
        
        /* Text area */
        .stTextArea > div > div > textarea {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
            caret-color: {bright_text} !important;
        }}
        
        .stTextArea label {{
            color: {text_color} !important;
        }}
        
        /* Multi-select */
        .stMultiSelect > div > div {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .stMultiSelect label {{
            color: {text_color} !important;
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
        
        /* Metric boxes */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: {bright_text} !important;
        }}
        
        [data-testid="stMetricDelta"] {{
            color: {text_color} !important;
        }}
        
        /* Dataframes */
        .dataframe {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
        }}
        
        .dataframe th {{
            background-color: {container_bg} !important;
            color: {bright_text} !important;
        }}
        
        .dataframe td {{
            color: {text_color} !important;
        }}
        
        /* Info/Warning boxes */
        .stAlert {{
            background-color: {input_bg} !important;
            border-color: {border_color} !important;
            color: {text_color} !important;
        }}
        
        .stAlert > div {{
            color: {text_color} !important;
            background-color: {input_bg} !important;
        }}
        
        /* Info boxes inside expanders */
        .streamlit-expanderContent .stAlert {{
            background-color: {input_bg} !important;
        }}
        
        .streamlit-expanderContent .stAlert * {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Markdown text */
        .stMarkdown {{
            color: {text_color} !important;
        }}
        
        .stMarkdown code {{
            background-color: {container_bg} !important;
            color: {bright_text} !important;
        }}
        
        /* Code blocks */
        .stCodeBlock {{
            background-color: {container_bg} !important;
        }}
        
        .stCodeBlock code {{
            color: {bright_text} !important;
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
        
        /* Spinner */
        .stSpinner > div {{
            border-color: {text_color} !important;
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
        </style>
        """, unsafe_allow_html=True)


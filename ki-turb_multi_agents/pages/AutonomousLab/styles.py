"""
CSS Styles for AI Assistant Page
Handles dark/light theme styling for chat interface
"""

import streamlit as st


def _inject_chat_toolbar_styles(*, is_dark: bool) -> None:
    """Compact Chat tools strip above compose (Copy / Export / Reset)."""
    if is_dark:
        eyebrow = "rgba(170, 200, 240, 0.8)"
        title = "#eef2f8"
        muted = "rgba(200, 210, 225, 0.7)"
        count_bg = "rgba(100, 150, 220, 0.18)"
        confirm_bg = "rgba(180, 70, 70, 0.12)"
        confirm_border = "rgba(220, 120, 120, 0.35)"
        dock_bg = "rgba(28, 32, 40, 0.92)"
        dock_border = "rgba(120, 160, 220, 0.22)"
    else:
        eyebrow = "rgba(50, 80, 120, 0.72)"
        title = "#152033"
        muted = "rgba(40, 55, 75, 0.62)"
        count_bg = "rgba(40, 90, 160, 0.08)"
        confirm_bg = "rgba(180, 50, 50, 0.06)"
        confirm_border = "rgba(180, 60, 60, 0.28)"
        dock_bg = "rgba(248, 250, 252, 0.96)"
        dock_border = "rgba(40, 70, 110, 0.14)"

    st.markdown(
        f"""
<style>
.lab-compose-actions {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin: 0.15rem 0 0.35rem 0;
  padding: 0.35rem 0.55rem;
  border: 1px solid {dock_border};
  border-radius: 10px;
  background: {dock_bg};
}}
.lab-compose-actions__label {{
  font-size: 0.66rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-weight: 650;
  color: {eyebrow};
}}
.lab-compose-actions__count {{
  font-size: 0.72rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
  background: {count_bg};
  color: {title};
}}
.lab-reset-confirm {{
  border: 1px solid {confirm_border};
  background: {confirm_bg};
  border-radius: 10px;
  padding: 0.55rem 0.7rem;
  margin: 0.35rem 0 0.45rem 0;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}}
.lab-reset-confirm strong {{
  color: {title};
  font-size: 0.84rem;
}}
.lab-reset-confirm span {{
  color: {muted};
  font-size: 0.78rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def inject_chat_styles():
    """Inject CSS styles for chat interface based on current theme"""
    theme_name = st.session_state.get("theme", "Light Scientific")
    is_dark = "Dark" in theme_name
    _inject_chat_toolbar_styles(is_dark=is_dark)

    if is_dark:
        from utils.theme_config import get_theme
        theme_info = get_theme(theme_name)
        bg_color = theme_info['paper_bgcolor']
        input_bg = "#3c3c3c"
        bright_text = "#ffffff"
        border_color = "#3e3e42"
        container_bg = "#2d2d30"
        
        st.markdown(f"""
        <style>
        /* Chat container styling */
        .stChatMessage {{
            padding: 1rem;
        }}
        
        /* Input area styling - completely merge with page */
        [data-testid="stForm"],
        [data-testid="stForm"] *,
        [data-testid="stForm"] * *,
        [data-testid="stForm"] * * * {{
            background: {bg_color} !important;
            background-color: {bg_color} !important;
        }}
        
        [data-testid="stForm"] {{
            position: sticky;
            bottom: 0;
            background: {bg_color} !important;
            background-color: {bg_color} !important;
            padding: 1rem 0;
            z-index: 100;
            margin: 0 !important;
            border: none !important;
        }}
        
        /* All possible Streamlit containers - force page background */
        [data-testid="stForm"] .element-container,
        [data-testid="stForm"] .block-container,
        [data-testid="stForm"] [data-testid="stVerticalBlock"],
        [data-testid="stForm"] [data-testid="stHorizontalBlock"],
        [data-testid="stForm"] [data-testid="stColumn"],
        [data-testid="stForm"] > div,
        [data-testid="stForm"] > div > div,
        [data-testid="stForm"] > div > div > div,
        [data-testid="stForm"] > div > div > div > div,
        /* Target columns that contain the input */
        [data-testid="column"]:has([data-testid="stForm"]),
        [data-testid="column"]:has(iframe[title*="multimodal"]),
        [data-testid="column"]:has(iframe[title*="chat_input"]),
        /* Target any div that might wrap the form */
        div:has([data-testid="stForm"]),
        div:has(iframe[title*="multimodal"]),
        div:has(iframe[title*="chat_input"]),
        /* Target parent containers */
        [data-testid="stForm"] ~ *,
        *:has([data-testid="stForm"]) {{
            background: {bg_color} !important;
            background-color: {bg_color} !important;
        }}
        
        /* Remove all card styling - completely flat */
        [data-testid="stForm"] > div,
        [data-testid="stForm"] > div > div,
        [data-testid="stForm"] > div > div > div {{
            background: {bg_color} !important;
            background-color: {bg_color} !important;
            border-radius: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Override ANY background color */
        [data-testid="stForm"] [style*="background"],
        [data-testid="stForm"] * [style*="background"],
        [data-testid="stForm"] * * [style*="background"] {{
            background: {bg_color} !important;
            background-color: {bg_color} !important;
        }}
        
        /* Multimodal input - only style actual inputs, not containers */
        [data-testid="stForm"] input,
        [data-testid="stForm"] textarea,
        [data-testid="stForm"] * input,
        [data-testid="stForm"] * textarea {{
            background-color: {input_bg} !important;
            color: {bright_text} !important;
            border-color: {border_color} !important;
        }}
        
        [data-testid="stForm"] button,
        [data-testid="stForm"] * button {{
            background-color: {container_bg} !important;
            color: {bright_text} !important;
        }}
        
        [data-testid="stForm"] svg,
        [data-testid="stForm"] * svg {{
            fill: {bright_text} !important;
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
        
        /* Dark surround for the multimodal component container - completely merge */
        div[data-testid="stIFrame"],
        div[data-testid="stIFrame"] *,
        div[data-testid="stIFrame"] * * {{
            background: {bg_color} !important;
            background-color: {bg_color} !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 0 !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        div[data-testid="stIFrame"] > div {{
            background: {bg_color} !important;
            background-color: {bg_color} !important;
        }}
        
        div[data-testid="stIFrame"] iframe {{
            background: transparent !important;
            border-radius: 0 !important;
        }}
        
        /* Target the parent element-container - completely merge */
        div.element-container:has(div[data-testid="stIFrame"] iframe[title*="chat_input_multimodal"]),
        div.element-container:has(div[data-testid="stIFrame"] iframe[src*="st_chat_input_multimodal"]),
        div.element-container:has(div[data-testid="stIFrame"] iframe[title*="multimodal"]),
        div.element-container:has([data-testid="stForm"]),
        div.element-container:has(iframe[title*="multimodal"]),
        div.element-container:has(iframe[title*="chat_input"]) {{
            background: {bg_color} !important;
            background-color: {bg_color} !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 0 !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Ensure main app canvas stays dark */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] > div,
        [data-testid="stMainBlockContainer"] > div > div {{
            background: {bg_color} !important;
        }}
        </style>
        <script>
        (function() {{
            const inputBg = '{input_bg}';
            const textColor = '{bright_text}';
            const borderColor = '{border_color}';
            const containerBg = '{container_bg}';
            const bgColor = '{bg_color}';
            
            function styleMultimodal() {{
                // Target form area and all children
                const form = document.querySelector('[data-testid="stForm"]');
                if (form) {{
                    // Force page background on form and ALL parents
                    form.style.setProperty('background-color', bgColor, 'important');
                    form.style.setProperty('background', bgColor, 'important');
                    
                    // Walk up the DOM tree and force page background on ALL parents
                    let parent = form.parentElement;
                    let depth = 0;
                    while (parent && parent !== document.body && depth < 10) {{
                        parent.style.setProperty('background-color', bgColor, 'important');
                        parent.style.setProperty('background', bgColor, 'important');
                        parent.style.setProperty('border-radius', '0', 'important');
                        parent.style.setProperty('padding', '0', 'important');
                        parent.style.setProperty('margin', '0', 'important');
                        parent.style.setProperty('box-shadow', 'none', 'important');
                        parent = parent.parentElement;
                        depth++;
                    }}
                    
                    // Force page background on ALL elements in form (except inputs/buttons)
                    form.querySelectorAll('*').forEach(el => {{
                        if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {{
                            el.style.setProperty('background-color', inputBg, 'important');
                            el.style.setProperty('color', textColor, 'important');
                            el.style.setProperty('border-color', borderColor, 'important');
                        }} else if (el.tagName === 'BUTTON') {{
                            el.style.setProperty('background-color', containerBg, 'important');
                            el.style.setProperty('color', textColor, 'important');
                        }} else if (el.tagName === 'SVG') {{
                            el.style.setProperty('fill', textColor, 'important');
                        }} else if (el.tagName === 'IFRAME') {{
                            // Skip iframes - they're handled separately
                        }} else {{
                            // Force page background on ALL containers
                            el.style.setProperty('background-color', bgColor, 'important');
                            el.style.setProperty('background', bgColor, 'important');
                            el.style.setProperty('border-radius', '0', 'important');
                            el.style.setProperty('box-shadow', 'none', 'important');
                        }}
                    }});
                    
                    // Also target iframe containers
                    document.querySelectorAll('[data-testid="stIFrame"]').forEach(iframeContainer => {{
                        iframeContainer.style.setProperty('background-color', bgColor, 'important');
                        iframeContainer.style.setProperty('background', bgColor, 'important');
                        iframeContainer.style.setProperty('padding', '0', 'important');
                        iframeContainer.style.setProperty('margin', '0', 'important');
                        iframeContainer.style.setProperty('border-radius', '0', 'important');
                    }});
                    
                    // Try iframe styling
                    form.querySelectorAll('iframe').forEach(iframe => {{
                        try {{
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            if (iframeDoc) {{
                                const style = iframeDoc.createElement('style');
                                style.textContent = `* {{ background: ${{inputBg}} !important; color: ${{textColor}} !important; }}`;
                                iframeDoc.head.appendChild(style);
                            }}
                        }} catch(e) {{}}
                    }});
                }}
            }}
            
            styleMultimodal();
            setTimeout(styleMultimodal, 100);
            setTimeout(styleMultimodal, 500);
            setTimeout(styleMultimodal, 1000);
            setTimeout(styleMultimodal, 2000);
            
            const observer = new MutationObserver(styleMultimodal);
            observer.observe(document.body, {{ childList: true, subtree: true }});
        }})();
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        /* Chat container styling */
        .stChatMessage {
            padding: 1rem;
        }
        
        /* Input area styling */
        [data-testid="stForm"] {
            position: sticky;
            bottom: 0;
            background-color: var(--background-color);
            padding: 1rem 0;
            z-index: 100;
        }
        </style>
        """, unsafe_allow_html=True)


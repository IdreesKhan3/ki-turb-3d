"""
CSS Styles for AI Assistant Page
Handles dark/light theme styling for chat interface
"""

import streamlit as st


def inject_chat_styles():
    """Inject CSS styles for chat interface based on current theme"""
    theme_name = st.session_state.get("theme", "Light Scientific")
    is_dark = "Dark" in theme_name
    
    if is_dark:
        from utils.theme_config import get_theme
        theme_info = get_theme(theme_name)
        bg_color = theme_info['paper_bgcolor']
        input_bg = "#3c3c3c"
        text_color = "#e8e8e8"
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


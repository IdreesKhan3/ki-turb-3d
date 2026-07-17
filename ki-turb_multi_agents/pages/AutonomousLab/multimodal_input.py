"""
Multimodal Input Handler
Handles text, voice, and image input using st-chat-input-multimodal component
"""

import streamlit as st
from typing import Optional, Dict, Any
import json


def render_multimodal_input(
    key: str = "chat_input",
    placeholder: str = "Ask anything, Explain, write, optimize, review the code",
    default_value: str = "",
    enabled_modes: list = ["text", "voice"],
    voice_enabled: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Render multimodal input component with text and voice support
    
    Args:
        key: Unique key for the component
        placeholder: Placeholder text for the input
        default_value: Default text value
        enabled_modes: List of enabled modes (text, voice, image)
        voice_enabled: Whether voice input is enabled
    
    Returns:
        Dictionary with input data if submitted, None otherwise
        Format: {
            "text": str,
            "audio": bytes (if voice input),
            "image": bytes (if image input),
            "mode": str ("text", "voice", or "image")
        }
    """
    try:
        # Try to import the multimodal component
        from st_chat_input_multimodal import multimodal_chat_input
        
        # Render the multimodal input
        # Note: The component uses enable_voice_input parameter
        result = multimodal_chat_input(
            key=key,
            placeholder=placeholder,
            enable_voice_input=voice_enabled,
            voice_recognition_method="web_speech",  # Uses browser's Web Speech API (no API key needed)
            max_recording_time=60,  # 60 seconds max recording
            disabled=False
        )
        
        # Process the result
        # Handle None or non-dict results safely
        if not result:
            return None
        
        if not isinstance(result, dict):
            # If result is not a dict, try to convert or return None
            return None
        
        # Result format from st-chat-input-multimodal:
        # - {"text": "user input", "files": [...], "audio_metadata": {...}}
        # - text: The input text (transcribed if voice, typed if text, or empty if image-only)
        # - files: List of uploaded files (images) - each file has "name" and "data" (base64)
        # - audio_metadata: Metadata about voice input if used (can be None or dict)
        
        # Safely extract text - handle None, empty string, or missing key
        # CRITICAL: Never block text input - always extract text if it exists
        text_value = result.get("text")
        if text_value is None:
            text_value = ""
        elif isinstance(text_value, str):
            text_value = text_value.strip()
        else:
            # Convert to string if it's not already
            text_value = str(text_value).strip() if text_value else ""
        
        input_data = {
            "text": text_value,  # Store text even if empty (for image-only uploads)
            "mode": "text"  # Default
        }
        
        # Safely handle audio_metadata (can be None)
        audio_metadata = result.get("audio_metadata")
        if audio_metadata and isinstance(audio_metadata, dict):
            if audio_metadata.get("used_voice_input", False):
                input_data["mode"] = "voice"
                # Store audio metadata if available
                if "audio_data" in audio_metadata:
                    input_data["audio"] = audio_metadata["audio_data"]
        
        # Check for image files
        files = result.get("files", [])
        if files and isinstance(files, list) and len(files) > 0:
            input_data["files"] = files
            # If image uploaded, set mode to image (unless voice was used)
            if input_data["mode"] == "text":
                input_data["mode"] = "image"
            # If both voice and image, keep as voice but note image presence
        
        # Return if there's text content OR image files (allow image-only uploads)
        # CRITICAL: Always return input_data if there's ANY text content
        # This ensures text input is NEVER blocked - text should always work!
        has_text_content = bool(input_data.get("text", "").strip())
        has_files = bool(files and isinstance(files, list) and len(files) > 0)
        
        # Return input_data if there's text OR files
        # Even if text is empty but files exist, return it (for image-only uploads)
        if has_text_content or has_files:
            return input_data
        
        # If no content at all (no text, no files), return None (user hasn't submitted yet)
        # This is correct - we shouldn't process empty submissions
        return None
        
    except ImportError:
        # Fallback to regular text input if component not installed
        st.warning("⚠️ Voice input not available. Install with: `pip install streamlit-chat-input-multimodal`")
        
        # Fallback to standard text input
        user_input = st.text_input(
            "💬 Ask me anything:",
            value=default_value,
            key=f"{key}_fallback",
            placeholder=placeholder,
            label_visibility="collapsed"
        )
        
        if user_input and user_input.strip():
            return {
                "text": user_input.strip(),
                "mode": "text"
            }
        
        return None
    
    except Exception as e:
        # Other errors
        st.error(f"Error with multimodal input: {str(e)}")
        st.info("💡 Falling back to text input. Install: `pip install streamlit-chat-input-multimodal`")
        
        # Fallback to standard text input
        user_input = st.text_input(
            "💬 Ask me anything:",
            value=default_value,
            key=f"{key}_fallback",
            placeholder=placeholder,
            label_visibility="collapsed"
        )
        
        if user_input and user_input.strip():
            return {
                "text": user_input.strip(),
                "mode": "text"
            }
        
        return None


def get_input_text(input_data: Optional[Dict[str, Any]]) -> str:
    """
    Extract text from multimodal input data
    
    Args:
        input_data: Dictionary from render_multimodal_input
    
    Returns:
        Text string (empty if no input)
    """
    if not input_data:
        return ""
    
    return input_data.get("text", "").strip()


def is_voice_input(input_data: Optional[Dict[str, Any]]) -> bool:
    """
    Check if input was from voice
    
    Args:
        input_data: Dictionary from render_multimodal_input
    
    Returns:
        True if voice input, False otherwise
    """
    if not input_data:
        return False
    
    return input_data.get("mode") == "voice"


def is_image_input(input_data: Optional[Dict[str, Any]]) -> bool:
    """
    Check if input was from image
    
    Args:
        input_data: Dictionary from render_multimodal_input
    
    Returns:
        True if image input, False otherwise
    """
    if not input_data:
        return False
    
    return "files" in input_data and len(input_data.get("files", [])) > 0

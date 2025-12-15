"""
Image Processor for LLM Vision Support
Handles conversion of uploaded images to formats compatible with vision-capable LLMs
Supports multiple LLM providers: Gemini, OpenAI, Anthropic, Qwen-VL, etc.
"""

import base64
import mimetypes
from typing import List, Dict, Any, Optional, Union
import streamlit as st


def _decode_base64_robust(base64_data: str) -> bytes:
    """
    Decode base64 string to bytes with padding fix and URL-safe support
    
    Args:
        base64_data: Base64 encoded string (standard or URL-safe)
    
    Returns:
        Decoded bytes
    
    Raises:
        ValueError: If decoding fails
    """
    # Remove whitespace
    base64_data = base64_data.strip()
    
    # Try standard base64 first
    try:
        # Fix padding if needed
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += '=' * (4 - missing_padding)
        return base64.b64decode(base64_data)
    except Exception:
        # Try URL-safe base64
        try:
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)
            return base64.urlsafe_b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64: {str(e)}")


def _infer_mime_type_from_filename(filename: str) -> str:
    """
    Infer MIME type from filename using mimetypes module
    
    Args:
        filename: File name or path
    
    Returns:
        MIME type string (defaults to "image/png")
    """
    if not filename:
        return "image/png"
    
    mime_type, _ = mimetypes.guess_type(filename.lower())
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    
    # Fallback for common extensions not in mimetypes
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    ext_to_mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "svg": "image/svg+xml",
        "heic": "image/heic",
        "heif": "image/heif",
    }
    return ext_to_mime.get(ext, "image/png")


def _parse_data_uri(data_uri: str) -> tuple:
    """
    Parse data URI and extract MIME type and base64 data
    
    Args:
        data_uri: Data URI string (e.g., "data:image/png;base64,<data>")
    
    Returns:
        Tuple of (mime_type, base64_data)
    
    Raises:
        ValueError: If data URI format is invalid
    """
    if not data_uri.startswith("data:"):
        raise ValueError("Data URI must start with 'data:'")
    
    if "," not in data_uri:
        raise ValueError("Data URI must contain comma separator")
    
    header, base64_data = data_uri.split(",", 1)
    
    # Validate header format: "data:<mime>;base64" or "data:<mime>"
    if not header.startswith("data:"):
        raise ValueError("Invalid data URI header")
    
    # Extract MIME type
    mime_part = header[5:]  # Remove "data:" prefix
    if ";base64" in mime_part:
        mime_type = mime_part.split(";base64")[0]
    else:
        mime_type = mime_part.split(";")[0] if ";" in mime_part else mime_part
    
    # Validate MIME type
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Invalid or non-image MIME type: {mime_type}")
    
    return mime_type, base64_data


def extract_images_from_session() -> Optional[List[Dict[str, Any]]]:
    """
    Extract uploaded images from session state (provider-agnostic format)
    
    Returns:
        List of image dictionaries with 'mime_type' and 'data' (bytes), or None
        Format: [{"mime_type": "image/png", "data": b"..."}, ...]
        Note: base64_data is only stored if needed for specific providers
    """
    if not hasattr(st, 'session_state'):
        return None
    
    image_files = st.session_state.get("_last_image_upload")
    if not image_files or not isinstance(image_files, list):
        return None
    
    processed_images = []
    for file in image_files:
        if not isinstance(file, dict):
            continue
        
        file_data = file.get("data", "")
        if not file_data:
            continue
        
        mime_type = None
        base64_data = None
        image_bytes = None
        
        # Parse data URI or raw base64
        if "," in file_data and file_data.startswith("data:"):
            try:
                mime_type, base64_data = _parse_data_uri(file_data)
            except ValueError as e:
                # Log warning but continue processing
                try:
                    st.warning(f"Invalid data URI format, skipping image: {str(e)}")
                except:
                    pass
                continue
        else:
            # Raw base64 data (no data URI header)
            base64_data = file_data
            # Infer MIME type from filename
            file_name = file.get("name", "")
            mime_type = _infer_mime_type_from_filename(file_name)
        
        # Decode base64 to bytes
        try:
            image_bytes = _decode_base64_robust(base64_data)
        except ValueError as e:
            # Log warning but continue processing
            try:
                st.warning(f"Failed to decode image, skipping: {str(e)}")
            except:
                pass
            continue
        
        # Store image
        # For Gemini, we only need bytes, but we store base64_data for providers that need it
        # to avoid re-encoding (memory trade-off: store both for efficiency)
        img_dict = {
            "mime_type": mime_type,
            "data": image_bytes,
        }
        # Store base64_data for providers that need it (OpenAI, Anthropic, etc.)
        # This avoids re-encoding but uses more memory
        if base64_data:
            img_dict["base64_data"] = base64_data
        processed_images.append(img_dict)
    
    return processed_images if processed_images else None


def _get_base64_data(img: Dict[str, Any]) -> str:
    """
    Get base64 data from image dict, encoding from bytes if needed
    
    Args:
        img: Image dict with 'data' (bytes) and optionally base64_data
    
    Returns:
        Base64 encoded string
    """
    # If we stored the original base64, use it (more efficient)
    if "base64_data" in img:
        return img["base64_data"]
    
    # Otherwise, encode from bytes
    image_bytes = img.get("data")
    if image_bytes:
        return base64.b64encode(image_bytes).decode('utf-8')
    
    return ""


def convert_to_provider_format(images: List[Dict[str, Any]], provider: str = "gemini") -> Union[List[Dict[str, Any]], List[str], Any]:
    """
    Convert images to the format required by a specific LLM provider
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
        provider: LLM provider name ('gemini', 'openai', 'anthropic', 'qwen-vl', etc.)
    
    Returns:
        Images in the format required by the specified provider
    """
    if not images:
        return []
    
    if provider.lower() == "gemini":
        return convert_to_gemini_format(images)
    elif provider.lower() == "openai":
        return convert_to_openai_format(images)
    elif provider.lower() == "anthropic":
        return convert_to_anthropic_format(images)
    elif provider.lower() in ["qwen-vl", "qwen"]:
        return convert_to_qwen_vl_format(images)
    else:
        # Default: return in universal format (mime_type + data)
        return convert_to_gemini_format(images)


def convert_to_gemini_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to Gemini API format
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in Gemini format: [{"mime_type": "...", "data": b"..."}]
    """
    gemini_images = []
    for img in images:
        gemini_images.append({
            "mime_type": img.get("mime_type", "image/png"),
            "data": img.get("data")
        })
    return gemini_images


def convert_to_openai_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to OpenAI GPT-4 Vision API format
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in OpenAI format: [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
    """
    openai_images = []
    for img in images:
        mime_type = img.get("mime_type", "image/png")
        base64_data = _get_base64_data(img)
        
        data_uri = f"data:{mime_type};base64,{base64_data}"
        openai_images.append({
            "type": "image_url",
            "image_url": {
                "url": data_uri
            }
        })
    return openai_images


def convert_to_anthropic_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to Anthropic Claude API format
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in Anthropic format: [{"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}]
    """
    anthropic_images = []
    for img in images:
        mime_type = img.get("mime_type", "image/png")
        base64_data = _get_base64_data(img)
        
        anthropic_images.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64_data
            }
        })
    return anthropic_images


def convert_to_qwen_vl_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to Qwen-VL API format (for future support)
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in Qwen-VL format (format may vary based on API)
    """
    qwen_images = []
    for img in images:
        mime_type = img.get("mime_type", "image/png")
        base64_data = _get_base64_data(img)
        
        qwen_images.append({
            "type": "image",
            "image": base64_data,
            "mime_type": mime_type
        })
    return qwen_images


def has_images_in_session() -> bool:
    """
    Check if there are images in session state
    
    Returns:
        True if images are available, False otherwise
    """
    if not hasattr(st, 'session_state'):
        return False
    
    image_files = st.session_state.get("_last_image_upload")
    return bool(image_files and isinstance(image_files, list) and len(image_files) > 0)


def clear_images_from_session():
    """Clear uploaded images from session state"""
    if hasattr(st, 'session_state'):
        if "_last_image_upload" in st.session_state:
            del st.session_state["_last_image_upload"]
